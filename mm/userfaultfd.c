// SPDX-License-Identifier: GPL-2.0-only
/*
 *  mm/userfaultfd.c
 *
 *  Copyright (C) 2015  Red Hat, Inc.
 */

#include <linux/mm.h>
#include <linux/sched/signal.h>
#include <linux/pagemap.h>
#include <linux/rmap.h>
#include <linux/swap.h>
#include <linux/swapops.h>
#include <linux/userfaultfd_k.h>
#include <linux/mmu_notifier.h>
#include <linux/hugetlb.h>
#include <linux/shmem_fs.h>
#include <asm/tlbflush.h>
#include "internal.h"

static __always_inline
struct vm_area_struct *find_dst_vma(struct mm_struct *dst_mm,
				    unsigned long dst_start,
				    unsigned long len)
{
	/*
	 * Make sure that the dst range is both valid and fully within a
	 * single existing vma.
	 */
	struct vm_area_struct *dst_vma;

	dst_vma = find_vma(dst_mm, dst_start);
	if (!dst_vma)
		return NULL;

	if (dst_start < dst_vma->vm_start ||
	    dst_start + len > dst_vma->vm_end)
		return NULL;

	/*
	 * Check the vma is registered in uffd, this is required to
	 * enforce the VM_MAYWRITE check done at uffd registration
	 * time.
	 */
	if (!rcu_access_pointer(dst_vma->vm_userfaultfd_ctx.ctx))
		return NULL;

	return dst_vma;
}

/*
 * Install PTEs, to map dst_addr (within dst_vma) to page.
 *
 * This function handles both MCOPY_ATOMIC_NORMAL and _CONTINUE for both shmem
 * and anon, and for both shared and private VMAs.
 */
int mfill_atomic_install_pte(struct mm_struct *dst_mm, pmd_t *dst_pmd,
			     struct vm_area_struct *dst_vma,
			     unsigned long dst_addr, struct page *page,
			     bool newly_allocated, bool wp_copy)
{
	int ret;
	pte_t _dst_pte, *dst_pte;
	bool writable = dst_vma->vm_flags & VM_WRITE;
	bool vm_shared = dst_vma->vm_flags & VM_SHARED;
	bool page_in_cache = page_mapping(page);
	spinlock_t *ptl;
	struct inode *inode;
	pgoff_t offset, max_off;

	_dst_pte = mk_pte(page, dst_vma->vm_page_prot);
	if (page_in_cache && !vm_shared)
		writable = false;
	if (writable || !page_in_cache)
		_dst_pte = pte_mkdirty(_dst_pte);
	if (writable) {
		if (wp_copy)
			_dst_pte = pte_mkuffd_wp(_dst_pte);
		else
			_dst_pte = pte_mkwrite(_dst_pte);
	}

	dst_pte = pte_offset_map_lock(dst_mm, dst_pmd, dst_addr, &ptl);

	if (vma_is_shmem(dst_vma)) {
		/* serialize against truncate with the page table lock */
		inode = dst_vma->vm_file->f_inode;
		offset = linear_page_index(dst_vma, dst_addr);
		max_off = DIV_ROUND_UP(i_size_read(inode), PAGE_SIZE);
		ret = -EFAULT;
		if (unlikely(offset >= max_off))
			goto out_unlock;
	}

	ret = -EEXIST;
	if (!pte_none(*dst_pte))
		goto out_unlock;

	if (page_in_cache)
		page_add_file_rmap(page, false);
	else
		page_add_new_anon_rmap(page, dst_vma, dst_addr, false);

	/*
	 * Must happen after rmap, as mm_counter() checks mapping (via
	 * PageAnon()), which is set by __page_set_anon_rmap().
	 */
	inc_mm_counter(dst_mm, mm_counter(page));

	if (newly_allocated)
		lru_cache_add_inactive_or_unevictable(page, dst_vma);

	set_pte_at(dst_mm, dst_addr, dst_pte, _dst_pte);

	/* No need to invalidate - it was non-present before */
	update_mmu_cache(dst_vma, dst_addr, dst_pte);
	ret = 0;
out_unlock:
	pte_unmap_unlock(dst_pte, ptl);
	return ret;
}

static int mcopy_atomic_pte(struct mm_struct *dst_mm,
			    pmd_t *dst_pmd,
			    struct vm_area_struct *dst_vma,
			    unsigned long dst_addr,
			    unsigned long src_addr,
			    struct page **pagep,
			    bool wp_copy)
{
	void *page_kaddr;
	int ret;
	struct page *page;

	if (!*pagep) {
		ret = -ENOMEM;
		page = alloc_page_vma(GFP_HIGHUSER_MOVABLE, dst_vma, dst_addr);
		if (!page)
			goto out;

		page_kaddr = kmap_local_page(page);
		/*
		 * The read mmap_lock is held here.  Despite the
		 * mmap_lock being read recursive a deadlock is still
		 * possible if a writer has taken a lock.  For example:
		 *
		 * process A thread 1 takes read lock on own mmap_lock
		 * process A thread 2 calls mmap, blocks taking write lock
		 * process B thread 1 takes page fault, read lock on own mmap lock
		 * process B thread 2 calls mmap, blocks taking write lock
		 * process A thread 1 blocks taking read lock on process B
		 * process B thread 1 blocks taking read lock on process A
		 *
		 * Disable page faults to prevent potential deadlock
		 * and retry the copy outside the mmap_lock.
		 */
		pagefault_disable();
		ret = copy_from_user(page_kaddr,
				     (const void __user *) src_addr,
				     PAGE_SIZE);
		pagefault_enable();
		kunmap_local(page_kaddr);

		/* fallback to copy_from_user outside mmap_lock */
		if (unlikely(ret)) {
			ret = -ENOENT;
			*pagep = page;
			/* don't free the page */
			goto out;
		}

		flush_dcache_page(page);
	} else {
		page = *pagep;
		*pagep = NULL;
	}

	/*
	 * The memory barrier inside __SetPageUptodate makes sure that
	 * preceding stores to the page contents become visible before
	 * the set_pte_at() write.
	 */
	__SetPageUptodate(page);

	ret = -ENOMEM;
	if (mem_cgroup_charge(page, dst_mm, GFP_KERNEL))
		goto out_release;

	ret = mfill_atomic_install_pte(dst_mm, dst_pmd, dst_vma, dst_addr,
				       page, true, wp_copy);
	if (ret)
		goto out_release;
out:
	return ret;
out_release:
	put_page(page);
	goto out;
}

static int mfill_zeropage_pte(struct mm_struct *dst_mm,
			      pmd_t *dst_pmd,
			      struct vm_area_struct *dst_vma,
			      unsigned long dst_addr)
{
	pte_t _dst_pte, *dst_pte;
	spinlock_t *ptl;
	int ret;
	pgoff_t offset, max_off;
	struct inode *inode;

	_dst_pte = pte_mkspecial(pfn_pte(my_zero_pfn(dst_addr),
					 dst_vma->vm_page_prot));
	dst_pte = pte_offset_map_lock(dst_mm, dst_pmd, dst_addr, &ptl);
	if (dst_vma->vm_file) {
		/* the shmem MAP_PRIVATE case requires checking the i_size */
		inode = dst_vma->vm_file->f_inode;
		offset = linear_page_index(dst_vma, dst_addr);
		max_off = DIV_ROUND_UP(i_size_read(inode), PAGE_SIZE);
		ret = -EFAULT;
		if (unlikely(offset >= max_off))
			goto out_unlock;
	}
	ret = -EEXIST;
	if (!pte_none(*dst_pte))
		goto out_unlock;
	set_pte_at(dst_mm, dst_addr, dst_pte, _dst_pte);
	/* No need to invalidate - it was non-present before */
	update_mmu_cache(dst_vma, dst_addr, dst_pte);
	ret = 0;
out_unlock:
	pte_unmap_unlock(dst_pte, ptl);
	return ret;
}

/* Handles UFFDIO_CONTINUE for all shmem VMAs (shared or private). */
static int mcontinue_atomic_pte(struct mm_struct *dst_mm,
				pmd_t *dst_pmd,
				struct vm_area_struct *dst_vma,
				unsigned long dst_addr,
				bool wp_copy)
{
	struct inode *inode = file_inode(dst_vma->vm_file);
	pgoff_t pgoff = linear_page_index(dst_vma, dst_addr);
	struct page *page;
	int ret;

	ret = shmem_getpage(inode, pgoff, &page, SGP_READ);
	if (ret)
		goto out;
	if (!page) {
		ret = -EFAULT;
		goto out;
	}

	ret = mfill_atomic_install_pte(dst_mm, dst_pmd, dst_vma, dst_addr,
				       page, false, wp_copy);
	if (ret)
		goto out_release;

	unlock_page(page);
	ret = 0;
out:
	return ret;
out_release:
	unlock_page(page);
	put_page(page);
	goto out;
}

static pmd_t *mm_alloc_pmd(struct mm_struct *mm, unsigned long address)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud;

	pgd = pgd_offset(mm, address);
	p4d = p4d_alloc(mm, pgd, address);
	if (!p4d)
		return NULL;
	pud = pud_alloc(mm, p4d, address);
	if (!pud)
		return NULL;
	/*
	 * Note that we didn't run this because the pmd was
	 * missing, the *pmd may be already established and in
	 * turn it may also be a trans_huge_pmd.
	 */
	return pmd_alloc(mm, pud, address);
}

#ifdef CONFIG_HUGETLB_PAGE
/*
 * __mcopy_atomic processing for HUGETLB vmas.  Note that this routine is
 * called with mmap_lock held, it will release mmap_lock before returning.
 */
static __always_inline ssize_t __mcopy_atomic_hugetlb(struct mm_struct *dst_mm,
					      struct vm_area_struct *dst_vma,
					      unsigned long dst_start,
					      unsigned long src_start,
					      unsigned long len,
					      bool *mmap_changing,
					      enum mcopy_atomic_mode mode)
{
	int vm_alloc_shared = dst_vma->vm_flags & VM_SHARED;
	int vm_shared = dst_vma->vm_flags & VM_SHARED;
	ssize_t err;
	pte_t *dst_pte;
	unsigned long src_addr, dst_addr;
	long copied;
	struct page *page;
	unsigned long vma_hpagesize;
	pgoff_t idx;
	u32 hash;
	struct address_space *mapping;

	/*
	 * There is no default zero huge page for all huge page sizes as
	 * supported by hugetlb.  A PMD_SIZE huge pages may exist as used
	 * by THP.  Since we can not reliably insert a zero page, this
	 * feature is not supported.
	 */
	if (mode == MCOPY_ATOMIC_ZEROPAGE) {
		mmap_read_unlock(dst_mm);
		return -EINVAL;
	}

	src_addr = src_start;
	dst_addr = dst_start;
	copied = 0;
	page = NULL;
	vma_hpagesize = vma_kernel_pagesize(dst_vma);

	/*
	 * Validate alignment based on huge page size
	 */
	err = -EINVAL;
	if (dst_start & (vma_hpagesize - 1) || len & (vma_hpagesize - 1))
		goto out_unlock;

retry:
	/*
	 * On routine entry dst_vma is set.  If we had to drop mmap_lock and
	 * retry, dst_vma will be set to NULL and we must lookup again.
	 */
	if (!dst_vma) {
		err = -ENOENT;
		dst_vma = find_dst_vma(dst_mm, dst_start, len);
		if (!dst_vma || !is_vm_hugetlb_page(dst_vma))
			goto out_unlock;

		err = -EINVAL;
		if (vma_hpagesize != vma_kernel_pagesize(dst_vma))
			goto out_unlock;

		vm_shared = dst_vma->vm_flags & VM_SHARED;
	}

	/*
	 * If not shared, ensure the dst_vma has a anon_vma.
	 */
	err = -ENOMEM;
	if (!vm_shared) {
		if (unlikely(anon_vma_prepare(dst_vma)))
			goto out_unlock;
	}

	while (src_addr < src_start + len) {
		BUG_ON(dst_addr >= dst_start + len);

		/*
		 * Serialize via i_mmap_rwsem and hugetlb_fault_mutex.
		 * i_mmap_rwsem ensures the dst_pte remains valid even
		 * in the case of shared pmds.  fault mutex prevents
		 * races with other faulting threads.
		 */
		mapping = dst_vma->vm_file->f_mapping;
		i_mmap_lock_read(mapping);
		idx = linear_page_index(dst_vma, dst_addr);
		hash = hugetlb_fault_mutex_hash(mapping, idx);
		mutex_lock(&hugetlb_fault_mutex_table[hash]);

		err = -ENOMEM;
		dst_pte = huge_pte_alloc(dst_mm, dst_vma, dst_addr, vma_hpagesize);
		if (!dst_pte) {
			mutex_unlock(&hugetlb_fault_mutex_table[hash]);
			i_mmap_unlock_read(mapping);
			goto out_unlock;
		}

		if (mode != MCOPY_ATOMIC_CONTINUE &&
		    !huge_pte_none(huge_ptep_get(dst_pte))) {
			err = -EEXIST;
			mutex_unlock(&hugetlb_fault_mutex_table[hash]);
			i_mmap_unlock_read(mapping);
			goto out_unlock;
		}

		err = hugetlb_mcopy_atomic_pte(dst_mm, dst_pte, dst_vma,
					       dst_addr, src_addr, mode, &page);

		mutex_unlock(&hugetlb_fault_mutex_table[hash]);
		i_mmap_unlock_read(mapping);
		vm_alloc_shared = vm_shared;

		cond_resched();

		if (unlikely(err == -ENOENT)) {
			mmap_read_unlock(dst_mm);
			BUG_ON(!page);

			err = copy_huge_page_from_user(page,
						(const void __user *)src_addr,
						vma_hpagesize / PAGE_SIZE,
						true);
			if (unlikely(err)) {
				err = -EFAULT;
				goto out;
			}
			mmap_read_lock(dst_mm);
			/*
			 * If memory mappings are changing because of non-cooperative
			 * operation (e.g. mremap) running in parallel, bail out and
			 * request the user to retry later
			 */
			if (mmap_changing && READ_ONCE(*mmap_changing)) {
				err = -EAGAIN;
				break;
			}

			dst_vma = NULL;
			goto retry;
		} else
			BUG_ON(page);

		if (!err) {
			dst_addr += vma_hpagesize;
			src_addr += vma_hpagesize;
			copied += vma_hpagesize;

			if (fatal_signal_pending(current))
				err = -EINTR;
		}
		if (err)
			break;
	}

out_unlock:
	mmap_read_unlock(dst_mm);
out:
	if (page) {
		/*
		 * We encountered an error and are about to free a newly
		 * allocated huge page.
		 *
		 * Reservation handling is very subtle, and is different for
		 * private and shared mappings.  See the routine
		 * restore_reserve_on_error for details.  Unfortunately, we
		 * can not call restore_reserve_on_error now as it would
		 * require holding mmap_lock.
		 *
		 * If a reservation for the page existed in the reservation
		 * map of a private mapping, the map was modified to indicate
		 * the reservation was consumed when the page was allocated.
		 * We clear the PagePrivate flag now so that the global
		 * reserve count will not be incremented in free_huge_page.
		 * The reservation map will still indicate the reservation
		 * was consumed and possibly prevent later page allocation.
		 * This is better than leaking a global reservation.  If no
		 * reservation existed, it is still safe to clear PagePrivate
		 * as no adjustments to reservation counts were made during
		 * allocation.
		 *
		 * The reservation map for shared mappings indicates which
		 * pages have reservations.  When a huge page is allocated
		 * for an address with a reservation, no change is made to
		 * the reserve map.  In this case PagePrivate will be set
		 * to indicate that the global reservation count should be
		 * incremented when the page is freed.  This is the desired
		 * behavior.  However, when a huge page is allocated for an
		 * address without a reservation a reservation entry is added
		 * to the reservation map, and PagePrivate will not be set.
		 * When the page is freed, the global reserve count will NOT
		 * be incremented and it will appear as though we have leaked
		 * reserved page.  In this case, set PagePrivate so that the
		 * global reserve count will be incremented to match the
		 * reservation map entry which was created.
		 *
		 * Note that vm_alloc_shared is based on the flags of the vma
		 * for which the page was originally allocated.  dst_vma could
		 * be different or NULL on error.
		 */
		if (vm_alloc_shared)
			SetPagePrivate(page);
		else
			ClearPagePrivate(page);
		put_page(page);
	}
	BUG_ON(copied < 0);
	BUG_ON(err > 0);
	BUG_ON(!copied && !err);
	return copied ? copied : err;
}
#else /* !CONFIG_HUGETLB_PAGE */
/* fail at build time if gcc attempts to use this */
extern ssize_t __mcopy_atomic_hugetlb(struct mm_struct *dst_mm,
				      struct vm_area_struct *dst_vma,
				      unsigned long dst_start,
				      unsigned long src_start,
				      unsigned long len,
				      bool *mmap_changing,
				      enum mcopy_atomic_mode mode);
#endif /* CONFIG_HUGETLB_PAGE */

static __always_inline ssize_t mfill_atomic_pte(struct mm_struct *dst_mm,
						pmd_t *dst_pmd,
						struct vm_area_struct *dst_vma,
						unsigned long dst_addr,
						unsigned long src_addr,
						struct page **page,
						enum mcopy_atomic_mode mode,
						bool wp_copy)
{
	ssize_t err;

	if (mode == MCOPY_ATOMIC_CONTINUE) {
		return mcontinue_atomic_pte(dst_mm, dst_pmd, dst_vma, dst_addr,
					    wp_copy);
	}

	/*
	 * The normal page fault path for a shmem will invoke the
	 * fault, fill the hole in the file and COW it right away. The
	 * result generates plain anonymous memory. So when we are
	 * asked to fill an hole in a MAP_PRIVATE shmem mapping, we'll
	 * generate anonymous memory directly without actually filling
	 * the hole. For the MAP_PRIVATE case the robustness check
	 * only happens in the pagetable (to verify it's still none)
	 * and not in the radix tree.
	 */
	if (!(dst_vma->vm_flags & VM_SHARED)) {
		if (mode == MCOPY_ATOMIC_NORMAL)
			err = mcopy_atomic_pte(dst_mm, dst_pmd, dst_vma,
					       dst_addr, src_addr, page,
					       wp_copy);
		else
			err = mfill_zeropage_pte(dst_mm, dst_pmd,
						 dst_vma, dst_addr);
	} else {
		VM_WARN_ON_ONCE(wp_copy);
		err = shmem_mfill_atomic_pte(dst_mm, dst_pmd, dst_vma,
					     dst_addr, src_addr,
					     mode != MCOPY_ATOMIC_NORMAL,
					     page);
	}

	return err;
}

static __always_inline ssize_t __mcopy_atomic(struct mm_struct *dst_mm,
					      unsigned long dst_start,
					      unsigned long src_start,
					      unsigned long len,
					      enum mcopy_atomic_mode mcopy_mode,
					      bool *mmap_changing,
					      __u64 mode)
{
	struct vm_area_struct *dst_vma;
	ssize_t err;
	pmd_t *dst_pmd;
	unsigned long src_addr, dst_addr;
	long copied;
	struct page *page;
	bool wp_copy;
#ifdef CONFIG_USERFAULTFD_DEBUG_LOG
	unsigned long last_dst = dst_start;
	unsigned long last_src = src_start;
#else
	unsigned long last_dst = 0;
	unsigned long last_src = 0;
#endif

#ifdef CONFIG_USERFAULTFD_DEBUG_LOG
#define UFFD_COPY_LOG(_fmt, ...)						\
	do {							\
		pr_err_ratelimited(_fmt, ##__VA_ARGS__);		\
	} while (0)
#else
#define UFFD_COPY_LOG(_fmt, ...)						\
	do {							\
	} while (0)
#endif

	/*
	 * Sanitize the command parameters:
	 */
	BUG_ON(dst_start & ~PAGE_MASK);
	BUG_ON(len & ~PAGE_MASK);

	/* Does the address range wrap, or is the span zero-sized? */
	BUG_ON(src_start + len <= src_start);
	BUG_ON(dst_start + len <= dst_start);

	src_addr = src_start;
	dst_addr = dst_start;
	copied = 0;
	page = NULL;
retry:
	err = -EAGAIN;
	if (mode & UFFDIO_MODE_MMAP_TRYLOCK) {
		if (!mmap_read_trylock(dst_mm)) {
			if (len == PAGE_SIZE) {
				/*
				 * ART uses single-page COPY as the MOVE fallback.
				 * Avoid returning zero-progress -EAGAIN for that
				 * path; it can otherwise retry the same page forever.
				 */
				UFFD_COPY_LOG("uffd_copy: __mcopy_atomic fallback reason=trylock_single_page_blocking mm=%px pid=%d tgid=%d dst=%#lx src=%#lx len=%#lx mode=%#llx copied=%ld\n",
					      dst_mm, current->pid, current->tgid,
					      dst_start, src_start, len,
					      (unsigned long long)mode, copied);
				mmap_read_lock(dst_mm);
			} else {
				UFFD_COPY_LOG("uffd_copy: __mcopy_atomic fail reason=%s mm=%px pid=%d tgid=%d dst=%#lx src=%#lx len=%#lx mode=%#llx copied=%ld ret=%d\n",
					      copied ? "trylock_eagain_after_progress" :
						       "trylock_eagain_zero_progress",
					      dst_mm, current->pid, current->tgid,
					      dst_start, src_start, len,
					      (unsigned long long)mode, copied, -EAGAIN);
				goto out;
			}
		}
	} else {
		mmap_read_lock(dst_mm);
	}

	/*
	 * If memory mappings are changing because of non-cooperative
	 * operation (e.g. mremap) running in parallel, bail out and
	 * request the user to retry later
	 */
	if (mmap_changing && READ_ONCE(*mmap_changing)) {
		UFFD_COPY_LOG("uffd_copy: __mcopy_atomic fail reason=mmap_changing mm=%px pid=%d tgid=%d dst=%#lx src=%#lx len=%#lx mode=%#llx copied=%ld ret=%d\n",
			      dst_mm, current->pid, current->tgid,
			      dst_start, src_start, len,
			      (unsigned long long)mode, copied, -EAGAIN);
		goto out_unlock;
	}

	/*
	 * Make sure the vma is not shared, that the dst range is
	 * both valid and fully within a single existing vma.
	 */
	err = -ENOENT;
	dst_vma = find_dst_vma(dst_mm, dst_start, len);
	if (!dst_vma)
		goto out_unlock;

	err = -EINVAL;
	/*
	 * shmem_zero_setup is invoked in mmap for MAP_ANONYMOUS|MAP_SHARED but
	 * it will overwrite vm_ops, so vma_is_anonymous must return false.
	 */
	if (WARN_ON_ONCE(vma_is_anonymous(dst_vma) &&
	    dst_vma->vm_flags & VM_SHARED))
		goto out_unlock;

	/*
	 * validate 'mode' now that we know the dst_vma: don't allow
	 * a wrprotect copy if the userfaultfd didn't register as WP.
	 */
	wp_copy = mode & UFFDIO_COPY_MODE_WP;
	if (wp_copy && !(dst_vma->vm_flags & VM_UFFD_WP))
		goto out_unlock;

	/*
	 * If this is a HUGETLB vma, pass off to appropriate routine
	 */
	if (is_vm_hugetlb_page(dst_vma))
		return  __mcopy_atomic_hugetlb(dst_mm, dst_vma, dst_start,
					       src_start, len, mmap_changing,
					       mcopy_mode);

	if (!vma_is_anonymous(dst_vma) && !vma_is_shmem(dst_vma))
		goto out_unlock;
	if (!vma_is_shmem(dst_vma) && mcopy_mode == MCOPY_ATOMIC_CONTINUE)
		goto out_unlock;

	/*
	 * Ensure the dst_vma has a anon_vma or this page
	 * would get a NULL anon_vma when moved in the
	 * dst_vma.
	 */
	err = -ENOMEM;
	if (!(dst_vma->vm_flags & VM_SHARED) &&
	    unlikely(anon_vma_prepare(dst_vma)))
		goto out_unlock;

	while (src_addr < src_start + len) {
		pmd_t dst_pmdval;

		BUG_ON(dst_addr >= dst_start + len);

		dst_pmd = mm_alloc_pmd(dst_mm, dst_addr);
		if (unlikely(!dst_pmd)) {
			err = -ENOMEM;
			break;
		}

		dst_pmdval = pmd_read_atomic(dst_pmd);
		/*
		 * If the dst_pmd is mapped as THP don't
		 * override it and just be strict.
		 */
		if (unlikely(pmd_trans_huge(dst_pmdval))) {
			err = -EEXIST;
			break;
		}
		if (unlikely(pmd_none(dst_pmdval)) &&
		    unlikely(__pte_alloc(dst_mm, dst_pmd))) {
			err = -ENOMEM;
			break;
		}
		/* If an huge pmd materialized from under us fail */
		if (unlikely(pmd_trans_huge(*dst_pmd))) {
			err = -EFAULT;
			break;
		}

		BUG_ON(pmd_none(*dst_pmd));
		BUG_ON(pmd_trans_huge(*dst_pmd));

		err = mfill_atomic_pte(dst_mm, dst_pmd, dst_vma, dst_addr,
				       src_addr, &page, mcopy_mode, wp_copy);
		cond_resched();

		if (unlikely(err == -ENOENT)) {
			void *page_kaddr;

			/*
			 * Return early due to mmap_lock contention only after
			 * some pages are copied to ensure that jank sensitive
			 * threads don't keep retrying for progress-critical
			 * pages.
			 */
			if (copied && mmap_lock_is_contended(dst_mm))
				break;

			mmap_read_unlock(dst_mm);
			BUG_ON(!page);

			page_kaddr = kmap_local_page(page);
			err = copy_from_user(page_kaddr,
					     (const void __user *) src_addr,
					     PAGE_SIZE);
			kunmap_local(page_kaddr);
			if (unlikely(err)) {
				err = -EFAULT;
				goto out;
			}
			flush_dcache_page(page);
			goto retry;
		} else
			BUG_ON(page);

		if (!err) {
			dst_addr += PAGE_SIZE;
			src_addr += PAGE_SIZE;
			copied += PAGE_SIZE;
			last_dst = dst_addr - PAGE_SIZE;
			last_src = src_addr - PAGE_SIZE;

			if (fatal_signal_pending(current))
				err = -EINTR;

			if (mmap_lock_is_contended(dst_mm)) {
				err = -EAGAIN;
				UFFD_COPY_LOG("uffd_copy: __mcopy_atomic fail reason=contention_after_progress mm=%px pid=%d tgid=%d dst=%#lx src=%#lx len=%#lx mode=%#llx copied=%ld last_dst=%#lx last_src=%#lx ret=%d\n",
					      dst_mm, current->pid, current->tgid,
					      dst_start, src_start, len,
					      (unsigned long long)mode, copied,
					      last_dst, last_src, -EAGAIN);
			}
		}
		if (err)
			break;
	}

out_unlock:
	mmap_read_unlock(dst_mm);
out:
	if (page)
		put_page(page);
	BUG_ON(copied < 0);
	BUG_ON(err > 0);
	BUG_ON(!copied && !err);
#ifdef CONFIG_USERFAULTFD_DEBUG_LOG
	if (err || (unsigned long)copied != len) {
		ssize_t returned = copied ? copied : err;

		UFFD_COPY_LOG("uffd_copy: __mcopy_atomic exit mm=%px pid=%d tgid=%d dst=%#lx src=%#lx len=%#lx mode=%#llx copied=%ld err=%zd returned=%zd last_dst=%#lx last_src=%#lx\n",
			      dst_mm, current->pid, current->tgid,
			      dst_start, src_start, len,
			      (unsigned long long)mode, copied, err, returned,
			      last_dst, last_src);
	}
#endif
#undef UFFD_COPY_LOG
	return copied ? copied : err;
}

ssize_t mcopy_atomic(struct mm_struct *dst_mm, unsigned long dst_start,
		     unsigned long src_start, unsigned long len,
		     bool *mmap_changing, __u64 mode)
{
	return __mcopy_atomic(dst_mm, dst_start, src_start, len,
			      MCOPY_ATOMIC_NORMAL, mmap_changing, mode);
}

ssize_t mfill_zeropage(struct mm_struct *dst_mm, unsigned long start,
		       unsigned long len, bool *mmap_changing, __u64 mode)
{
	return __mcopy_atomic(dst_mm, start, 0, len, MCOPY_ATOMIC_ZEROPAGE,
			      mmap_changing, mode);
}

ssize_t mcopy_continue(struct mm_struct *dst_mm, unsigned long start,
		       unsigned long len, bool *mmap_changing)
{
	return __mcopy_atomic(dst_mm, start, 0, len, MCOPY_ATOMIC_CONTINUE,
			      mmap_changing, 0);
}

static inline void uffd_double_pt_lock(spinlock_t *ptl1, spinlock_t *ptl2)
{
	if (ptl1 > ptl2) {
		spinlock_t *tmp = ptl1;

		ptl1 = ptl2;
		ptl2 = tmp;
	}
	spin_lock(ptl1);
	if (ptl1 != ptl2)
		spin_lock_nested(ptl2, SINGLE_DEPTH_NESTING);
	else
		__acquire(ptl2);
}

static inline void uffd_double_pt_unlock(spinlock_t *ptl1, spinlock_t *ptl2)
{
	spin_unlock(ptl1);
	if (ptl1 != ptl2)
		spin_unlock(ptl2);
	else
		__release(ptl2);
}

static inline bool vma_move_compatible(struct vm_area_struct *vma)
{
	return !(vma->vm_flags & (VM_PFNMAP | VM_IO | VM_HUGETLB | VM_MIXEDMAP));
}

#define UFFD_MOVE_TRANSIENT_RETRIES 1

#ifdef CONFIG_SWAP
static int move_swap_pte(struct mm_struct *mm,
			 struct vm_area_struct *dst_vma,
			 struct vm_area_struct *src_vma,
			 pmd_t *dst_pmd, pmd_t *src_pmd,
			 unsigned long dst_addr, unsigned long src_addr,
			 pte_t orig_src_pte)
{
	swp_entry_t entry;
	struct swap_info_struct *si;
	struct page *page;
	unsigned long offset;
	pte_t *dst_pte = NULL, *src_pte = NULL;
	spinlock_t *dst_ptl, *src_ptl;
	pte_t moved_pte;
	int ret = -EBUSY;

	if (!dst_vma->anon_vma)
		return -EBUSY;

	if (!is_swap_pte(orig_src_pte))
		return -EBUSY;

	entry = pte_to_swp_entry(orig_src_pte);
	if (non_swap_entry(entry))
		return -EBUSY;

	si = get_swap_device(entry);
	if (!si)
		return -EBUSY;
	offset = swp_offset(entry);
	if (READ_ONCE(si->swap_map[offset]) & SWAP_HAS_CACHE)
		goto out_put_swap;

	if (swp_swapcount(entry) != 1)
		goto out_put_swap;

	page = lookup_swap_cache(entry, src_vma, src_addr);
	if (page) {
		put_page(page);
		goto out_put_swap;
	}

	if (READ_ONCE(si->swap_map[offset]) & SWAP_HAS_CACHE)
		goto out_put_swap;

	dst_pte = pte_offset_map(dst_pmd, dst_addr);
	src_pte = pte_offset_map(src_pmd, src_addr);
	if (!dst_pte || !src_pte) {
		ret = -EFAULT;
		goto out_unmap;
	}

	dst_ptl = pte_lockptr(mm, dst_pmd);
	src_ptl = pte_lockptr(mm, src_pmd);
	uffd_double_pt_lock(dst_ptl, src_ptl);

	if (!pte_same(*src_pte, orig_src_pte))
		goto out_unlock;
	if (!pte_none(*dst_pte))
		goto out_unlock;
	if (unlikely(READ_ONCE(si->swap_map[offset]) & SWAP_HAS_CACHE))
		goto out_unlock;

	moved_pte = ptep_get_and_clear(mm, src_addr, src_pte);
#ifdef CONFIG_MEM_SOFT_DIRTY
	moved_pte = pte_swp_mksoft_dirty(moved_pte);
#endif
	set_pte_at(mm, dst_addr, dst_pte, moved_pte);
	ret = 0;

out_unlock:
	uffd_double_pt_unlock(dst_ptl, src_ptl);
out_unmap:
	if (src_pte)
		pte_unmap(src_pte);
	if (dst_pte)
		pte_unmap(dst_pte);
out_put_swap:
	put_swap_device(si);
	return ret;
}

#else
static int move_swap_pte(struct mm_struct *mm,
			 struct vm_area_struct *dst_vma,
			 struct vm_area_struct *src_vma,
			 pmd_t *dst_pmd, pmd_t *src_pmd,
			 unsigned long dst_addr, unsigned long src_addr,
			 pte_t orig_src_pte)
{
	(void)mm;
	(void)dst_vma;
	(void)src_vma;
	(void)dst_pmd;
	(void)src_pmd;
	(void)dst_addr;
	(void)src_addr;
	(void)orig_src_pte;

	return -EBUSY;
}
#endif

static int validate_move_areas(struct userfaultfd_ctx *ctx,
			       struct vm_area_struct *src_vma,
			       struct vm_area_struct *dst_vma)
{
	if ((src_vma->vm_flags & VM_ACCESS_FLAGS) !=
	    (dst_vma->vm_flags & VM_ACCESS_FLAGS))
		return -EINVAL;

	if (pgprot_val(src_vma->vm_page_prot) !=
	    pgprot_val(dst_vma->vm_page_prot))
		return -EINVAL;

	if ((src_vma->vm_flags & VM_LOCKED) !=
	    (dst_vma->vm_flags & VM_LOCKED))
		return -EINVAL;

	if (!(src_vma->vm_flags & VM_WRITE))
		return -EINVAL;

	if (!vma_move_compatible(src_vma) ||
	    !vma_move_compatible(dst_vma))
		return -EINVAL;

	if (!dst_vma->vm_userfaultfd_ctx.ctx ||
	    dst_vma->vm_userfaultfd_ctx.ctx != ctx)
		return -EINVAL;

	if (!vma_is_anonymous(src_vma) || !vma_is_anonymous(dst_vma))
		return -EINVAL;

	return 0;
}

static ssize_t move_pages_pte(struct mm_struct *mm,
			      pmd_t *dst_pmd, pmd_t *src_pmd,
			      struct vm_area_struct *dst_vma,
			      struct vm_area_struct *src_vma,
			      unsigned long dst_addr, unsigned long src_addr,
			      __u64 mode,
			      bool *need_tlb_flush,
			      unsigned long *tlb_flush_start,
			      unsigned long *tlb_flush_end)
{
	pte_t *dst_pte, *src_pte;
	spinlock_t *dst_ptl, *src_ptl;
	pte_t orig_dst_pte, orig_src_pte, moved_pte;
	pte_t swap_src_pte = __pte(0);
	struct page *page, *locked_retry_page;
	unsigned int transient_retries;
	bool copied, retry_same, try_swap_pte, wait_page_retry;
	ssize_t err = 0;

	transient_retries = 0;
	locked_retry_page = NULL;

retry_same_page:
	dst_pte = NULL;
	src_pte = NULL;
	page = NULL;
	copied = false;
	retry_same = false;
	try_swap_pte = false;
	wait_page_retry = false;

	dst_pte = pte_offset_map(dst_pmd, dst_addr);
	src_pte = pte_offset_map(src_pmd, src_addr);
	if (unlikely(!dst_pte || !src_pte)) {
		err = -EFAULT;
		goto out_unmap;
	}

	dst_ptl = pte_lockptr(mm, dst_pmd);
	src_ptl = pte_lockptr(mm, src_pmd);
	uffd_double_pt_lock(dst_ptl, src_ptl);

	orig_dst_pte = *dst_pte;
	orig_src_pte = *src_pte;

	if (!pte_none(orig_dst_pte)) {
		err = -EEXIST;
		if (pte_present(orig_dst_pte) && pte_present(orig_src_pte) &&
		    pte_pfn(orig_dst_pte) != pte_pfn(orig_src_pte))
			err = -EBUSY;
		goto out_unlock_pt;
	}
	if (pte_none(orig_src_pte)) {
		if (mode & UFFDIO_MOVE_MODE_ALLOW_SRC_HOLES) {
			copied = true;
			goto out_unlock_pt;
		}
		err = -ENOENT;
		goto out_unlock_pt;
	}
	if (!pte_present(orig_src_pte)) {
		swap_src_pte = orig_src_pte;
		try_swap_pte = true;
		err = 0;
		goto out_unlock_pt;
	}

	if (is_zero_pfn(pte_pfn(orig_src_pte))) {
		pte_t zero_pte;

		ptep_clear_flush(src_vma, src_addr, src_pte);
		zero_pte = pte_mkspecial(pfn_pte(my_zero_pfn(dst_addr),
						 dst_vma->vm_page_prot));
		set_pte_at(mm, dst_addr, dst_pte, zero_pte);
		update_mmu_cache(dst_vma, dst_addr, dst_pte);
		copied = true;
		goto out_unlock_pt;
	}

	page = vm_normal_page(src_vma, src_addr, orig_src_pte);
	if (!page || !PageAnon(page) || PageCompound(page) ||
	    page_mapcount(page) != 1 || page_maybe_dma_pinned(page)) {
		err = -EBUSY;
		goto out_unlock_pt;
	}
	if (locked_retry_page) {
		if (page != locked_retry_page) {
			unlock_page(locked_retry_page);
			put_page(locked_retry_page);
			locked_retry_page = NULL;
			if (transient_retries < UFFD_MOVE_TRANSIENT_RETRIES) {
				transient_retries++;
				retry_same = true;
				err = 0;
				goto out_unlock_pt;
			}
			err = -EAGAIN;
			goto out_unlock_pt;
		}
	} else if (!trylock_page(page)) {
		if (transient_retries < UFFD_MOVE_TRANSIENT_RETRIES) {
			transient_retries++;
			get_page(page);
			locked_retry_page = page;
			wait_page_retry = true;
			err = 0;
			goto out_unlock_pt;
		}
		err = -EAGAIN;
		goto out_unlock_pt;
	}
	if (!pte_same(orig_src_pte, *src_pte) ||
	    !pte_same(orig_dst_pte, *dst_pte)) {
		unlock_page(page);
		if (locked_retry_page) {
			put_page(locked_retry_page);
			locked_retry_page = NULL;
		}
		if (transient_retries < UFFD_MOVE_TRANSIENT_RETRIES) {
			transient_retries++;
			retry_same = true;
			err = 0;
			goto out_unlock_pt;
		}
		err = -EAGAIN;
		goto out_unlock_pt;
	}

	moved_pte = ptep_get_and_clear(mm, src_addr, src_pte);
	if (unlikely(page_maybe_dma_pinned(page))) {
		set_pte_at(mm, src_addr, src_pte, moved_pte);
		unlock_page(page);
		if (locked_retry_page) {
			put_page(locked_retry_page);
			locked_retry_page = NULL;
		}
		err = -EBUSY;
		goto out_unlock_pt;
	}
	page_move_anon_rmap(page, dst_vma);
	page->index = linear_page_index(dst_vma, dst_addr);
	set_pte_at(mm, dst_addr, dst_pte, moved_pte);
	update_mmu_cache(dst_vma, dst_addr, dst_pte);
	unlock_page(page);
	if (locked_retry_page) {
		put_page(locked_retry_page);
		locked_retry_page = NULL;
	}
	if (!*need_tlb_flush) {
		*need_tlb_flush = true;
		*tlb_flush_start = src_addr;
	}
	*tlb_flush_end = src_addr + PAGE_SIZE;
	copied = true;

out_unlock_pt:
	uffd_double_pt_unlock(dst_ptl, src_ptl);
out_unmap:
	if (src_pte)
		pte_unmap(src_pte);
	if (dst_pte)
		pte_unmap(dst_pte);

	if (wait_page_retry) {
		err = lock_page_killable(locked_retry_page);
		if (err) {
			put_page(locked_retry_page);
			locked_retry_page = NULL;
			goto out_done;
		}
		goto retry_same_page;
	}
	if (retry_same) {
		cond_resched();
		goto retry_same_page;
	}
	if (locked_retry_page) {
		unlock_page(locked_retry_page);
		put_page(locked_retry_page);
		locked_retry_page = NULL;
	}

	if (try_swap_pte) {
		err = move_swap_pte(mm, dst_vma, src_vma, dst_pmd, src_pmd,
				    dst_addr, src_addr, swap_src_pte);
		if (!err)
			copied = true;
	}

	if (copied) {
		err = PAGE_SIZE;
		goto out_done;
	}

	if (unlikely(!err)) {
		err = -EFAULT;
	}

out_done:
	return err;
}


ssize_t move_pages(struct mm_struct *mm, struct userfaultfd_ctx *ctx,
		   unsigned long dst_start, unsigned long src_start,
		   unsigned long len, __u64 mode, bool *mmap_changing)
{
	struct vm_area_struct *dst_vma, *src_vma;
	unsigned long dst_addr, src_addr;
	ssize_t moved = 0, err = -EINVAL;
	ssize_t ret;
	unsigned long tlb_flush_start = 0, tlb_flush_end = 0;
	bool need_tlb_flush = false;

	BUG_ON(dst_start & ~PAGE_MASK);
	BUG_ON(src_start & ~PAGE_MASK);
	BUG_ON(len & ~PAGE_MASK);
	BUG_ON(dst_start + len <= dst_start);
	BUG_ON(src_start + len <= src_start);

	mmap_read_lock(mm);

	err = -EAGAIN;
	if (mmap_changing && READ_ONCE(*mmap_changing))
		goto out_unlock;

	err = -ENOENT;
	dst_vma = find_vma(mm, dst_start);
	src_vma = find_vma(mm, src_start);
	if (!dst_vma || !src_vma)
		goto out_unlock;

	if (dst_start < dst_vma->vm_start || dst_start + len > dst_vma->vm_end ||
	    src_start < src_vma->vm_start || src_start + len > src_vma->vm_end)
		goto out_unlock;

	err = validate_move_areas(ctx, src_vma, dst_vma);
	if (err)
		goto out_unlock;

	err = 0;
	for (dst_addr = dst_start, src_addr = src_start;
	     src_addr < src_start + len;
	     dst_addr += PAGE_SIZE, src_addr += PAGE_SIZE) {
		pmd_t *dst_pmd, *src_pmd;

		if (fatal_signal_pending(current)) {
			err = -EINTR;
			break;
		}

		dst_pmd = mm_alloc_pmd(mm, dst_addr);
		if (unlikely(!dst_pmd)) {
			err = -ENOMEM;
			break;
		}
		if (unlikely(pmd_trans_huge(*dst_pmd))) {
			err = -EEXIST;
			break;
		}

		src_pmd = mm_find_pmd(mm, src_addr);
		if (unlikely(!src_pmd)) {
			if (!(mode & UFFDIO_MOVE_MODE_ALLOW_SRC_HOLES)) {
				err = -ENOENT;
				break;
			}
			src_pmd = mm_alloc_pmd(mm, src_addr);
			if (unlikely(!src_pmd)) {
				err = -ENOMEM;
				break;
			}
		}
		if (pmd_none(*src_pmd)) {
			if (!(mode & UFFDIO_MOVE_MODE_ALLOW_SRC_HOLES)) {
				err = -ENOENT;
				break;
			}
			if (unlikely(__pte_alloc(mm, src_pmd))) {
				err = -ENOMEM;
				break;
			}
		}
		if (unlikely(pmd_trans_huge(*src_pmd))) {
			err = -EBUSY;
			break;
		}
		if (unlikely(pmd_none(*dst_pmd)) && unlikely(__pte_alloc(mm, dst_pmd))) {
			err = -ENOMEM;
			break;
		}

		err = move_pages_pte(mm, dst_pmd, src_pmd, dst_vma, src_vma,
				     dst_addr, src_addr, mode, &need_tlb_flush,
				     &tlb_flush_start, &tlb_flush_end);
		if (err == PAGE_SIZE) {
			moved += PAGE_SIZE;
			err = 0;
			continue;
		}
		break;
	}

	if (need_tlb_flush)
		flush_tlb_range(src_vma, tlb_flush_start, tlb_flush_end);

out_unlock:
	ret = moved ? moved : err;
	mmap_read_unlock(mm);
	return ret;
}

int mwriteprotect_range(struct mm_struct *dst_mm, unsigned long start,
			unsigned long len, bool enable_wp, bool *mmap_changing)
{
	struct vm_area_struct *dst_vma;
	pgprot_t newprot;
	int err;

	/*
	 * Sanitize the command parameters:
	 */
	BUG_ON(start & ~PAGE_MASK);
	BUG_ON(len & ~PAGE_MASK);

	/* Does the address range wrap, or is the span zero-sized? */
	BUG_ON(start + len <= start);

	mmap_read_lock(dst_mm);

	/*
	 * If memory mappings are changing because of non-cooperative
	 * operation (e.g. mremap) running in parallel, bail out and
	 * request the user to retry later
	 */
	err = -EAGAIN;
	if (mmap_changing && READ_ONCE(*mmap_changing))
		goto out_unlock;

	err = -ENOENT;
	dst_vma = find_dst_vma(dst_mm, start, len);
	/*
	 * Make sure the vma is not shared, that the dst range is
	 * both valid and fully within a single existing vma.
	 */
	if (!dst_vma || (dst_vma->vm_flags & VM_SHARED))
		goto out_unlock;
	if (!userfaultfd_wp(dst_vma))
		goto out_unlock;
	if (!vma_is_anonymous(dst_vma))
		goto out_unlock;

	if (enable_wp)
		newprot = vm_get_page_prot(dst_vma->vm_flags & ~(VM_WRITE));
	else
		newprot = vm_get_page_prot(dst_vma->vm_flags);

	change_protection(dst_vma, start, start + len, newprot,
			  enable_wp ? MM_CP_UFFD_WP : MM_CP_UFFD_WP_RESOLVE);

	err = 0;
out_unlock:
	mmap_read_unlock(dst_mm);
	return err;
}
