#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NUM_NODES (2 << 10)

typedef struct node {
    int val;
    struct node *next;
} node;

static node *build_list(int n) {
    node *head = NULL;
    node *tail = NULL;
    for (int i = 0; i < n; i++) {
        node *nd = malloc(sizeof(node));
        if (!nd) {
            perror("malloc");
            exit(1);
        }
        nd->val = i;
        nd->next = NULL;
        if (tail)
            tail->next = nd;
        else
            head = nd;
        tail = nd;
    }
    return head;
}

static void free_list(node *head) {
    while (head) {
        node *next = head->next;
        free(head);
        head = next;
    }
}

// Load each node's val through a volatile pointer so the compiler can't
// skip the memory access.
static void traverse(node *head) {
    for (node *cur = head; cur; cur = cur->next)
        (void)*(volatile int *)&cur->val;
}

static uint64_t time_traversal(node *head) {
    uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    traverse(head);
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start;
}

static int cmp_uintptr(const void *a, const void *b) {
    uintptr_t x = *(const uintptr_t *)a, y = *(const uintptr_t *)b;
    return (x > y) - (x < y);
}

// Report how the allocator spread the nodes across virtual pages.
static void page_stats(node *head) {
    uintptr_t page_size = (uintptr_t)getpagesize();
    uintptr_t pages[NUM_NODES];
    int n = 0;
    int crossings = 0;
    for (node *cur = head; cur; cur = cur->next) {
        pages[n++] = (uintptr_t)cur / page_size;
        if (cur->next && (uintptr_t)cur->next / page_size != pages[n - 1])
            crossings++;
    }

    qsort(pages, n, sizeof(pages[0]), cmp_uintptr);
    int distinct = 0, min_run = n, max_run = 0;
    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && pages[j] == pages[i])
            j++;
        int run = j - i;
        if (run < min_run)
            min_run = run;
        if (run > max_run)
            max_run = run;
        distinct++;
        i = j;
    }

    printf("page size: %lu bytes, node size: %zu bytes\n", page_size, sizeof(node));
    printf("first node: %p, pages spanned: %lu\n", (void *)head, pages[n - 1] - pages[0] + 1);
    printf("%d nodes on %d distinct pages\n", n, distinct);
    printf("nodes per page: min %d, max %d, avg %.1f\n", min_run, max_run, (double)n / distinct);
    printf("next pointers that cross a page: %d of %d\n", crossings, n - 1);
}

int main(void) {
    node *head = build_list(NUM_NODES);
    page_stats(head);

    uint64_t t1 = time_traversal(head);

    printf("traversal 1: %llu ns\n", t1);

    free_list(head);
    return 0;
}
