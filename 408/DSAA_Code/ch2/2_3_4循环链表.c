#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType int

typedef struct LNode{
    ElemType data;
    struct LNode* prior;
    struct LNode* next;
}LNode, *LinkList;


bool InitList(LinkList* L){
    *L = (LNode* )malloc(sizeof(LNode));
    (*L)->next = *L;
    (*L)->prior = *L;
    return true;
}


int Length(LinkList L){
    int len = 0;
    if(L == NULL)return -1;
    LNode* p = L;
    while(p->next != L){
        len++;
        p = p->next;
    }
    return len;
}


LNode* GetElem(LinkList L, int Index){
    if(L == NULL)return NULL;
    int len = Length(L);
    if(Index >= len || Index < 0){
        return NULL;
    }
    LNode* p = L;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    return p;
}

LNode* LocateElem(LinkList L, ElemType e){
    if(L->next == NULL)return NULL;
    LNode* p = L->next;
    while(p->next != L && p->data != e){
        p = p->next;
    }
    if (p->next == L && p->data != e)return NULL;
    return p;
}

// 尾插法
bool LinkInsertBack(LinkList L, int Index, ElemType e){
    if(L == NULL)return false;
    LNode* p = L;
    int len = Length(L);
    if(Index > len || Index < 0)return false;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    LNode* temp = (LNode* )malloc(sizeof(LNode));
    temp->data = e;
    temp->next = p->next;
    temp->prior = p;
    p->next->prior = temp;
    p->next = temp;
    return true;
}

// 前插法
bool LinkInsertFront(LinkList L, int Index, ElemType e){
    if(L == NULL)return false;
    LNode* p = L;
    int len = Length(L);
    if(Index > len || Index < 0)return false;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    LNode* temp = (LNode* )malloc(sizeof(LNode));
    temp->data = e;
    temp->next = p;
    temp->prior = p->prior;
    p->prior->next = temp;
    p->prior = temp;
    return true;
}


bool ListDelete(LinkList L, int Index, ElemType* e){
    if(L == NULL)return false;
    LNode* p = L;
    int len = Length(L);
    if(Index >= len || Index < 0)return false;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    *e = p->data;
    p->prior->next = p->next;
    p->next->prior = p->prior;
    free(p);
    return true;
}

// 头插法
LinkList List_HeadInsert(LinkList L){
    if(L == NULL)return NULL;
    LNode* temp;
    int x;
    scanf("%d", &x);
    while(x != 9999){
        temp = (LNode* )malloc(sizeof(LNode));
        temp->data = x;
        temp->next = L->next;
        temp->prior = L;
        L->next->prior = temp;
        L->next = temp;
        scanf("%d", &x);
    }
    return L;
}


// 尾插法
LinkList List_TailInsert(LinkList L){
    if(L == NULL)return NULL;
    LNode* temp;
    int x;
    scanf("%d", &x);
    while(x != 9999){
        temp = (LNode* )malloc(sizeof(LNode));
        temp->data = x;
        temp->next = L;
        temp->prior = L->prior;
        L->prior->next = temp;
        L->prior = temp;
        scanf("%d", &x);
    }
    return L;
}

bool DestroyList(LinkList L){
    if(L == NULL)return false;
    LNode* p = L->next;
    if(p == L){
        free(p);
        return true;
    }
    while(p != L){
        LNode* temp = p->next;
        free(p);
        p = temp;
    }
    free(p);
    p = NULL;
    return true;
}


void PrintList(LinkList L){
    if(L == NULL){
        printf("列表为空！");
        return;
    }
    LNode* p = L;
    while(p->next != L){
        p = p->next;
        printf("%d ", p->data);
    }
    return;
}

int main(){
    LinkList L;
    ElemType e;

    // ========== 1. 初始化 ==========
    printf("========== 1. 初始化链表 ==========\n");
    InitList(&L);
    printf("空表长度: %d\n", Length(L));
    PrintList(L);

    // ========== 2. 后插法建表: 1 2 3 4 5 ==========
    printf("\n========== 2. 后插法插入 1~5 ==========\n");
    LinkInsertBack(L, 0, 1);
    LinkInsertBack(L, 1, 2);
    LinkInsertBack(L, 2, 3);
    LinkInsertBack(L, 3, 4);
    LinkInsertBack(L, 4, 5);
    printf("长度: %d, 链表: ", Length(L));
    PrintList(L);

    // ========== 3. 按位查找 GetElem ==========
    printf("\n========== 3. 按位查找 ==========\n");
    for(int i = 0; i < 5; i++){
        LNode* node = GetElem(L, i);
        if(node) printf("第%d个元素: %d\n", i+1, node->data);
    }
    printf("越界查找(Index=5): %s\n", GetElem(L, 5) ? "BUG" : "NULL(正确)");

    // ========== 4. 按值查找 LocateElem ==========
    printf("\n========== 4. 按值查找 ==========\n");
    printf("查找3: %s\n", LocateElem(L, 3) ? "找到(正确)" : "未找到");
    printf("查找1(首结点): %s\n", LocateElem(L, 1) ? "找到(正确)" : "未找到");
    printf("查找5(尾结点): %s\n", LocateElem(L, 5) ? "找到(正确)" : "未找到");
    printf("查找999(不存在): %s\n", LocateElem(L, 999) ? "BUG" : "NULL(正确)");

    // ========== 5. 前插法: 在值为3的位置前插入99 ==========
    printf("\n========== 5. 前插法(在3前插入99) ==========\n");
    LinkInsertFront(L, 2, 99);   // Index=2 是值为3的结点, 在其前插入
    printf("插入后: ");
    PrintList(L);                 // 期望: 1 2 99 3 4 5

    // ========== 6. 后插法: 在值为3的位置后插入88 ==========
    printf("\n========== 6. 后插法(在3后插入88) ==========\n");
    LinkInsertBack(L, 3, 88);    // Index=3 现在是值为3的结点, 在其后插入
    printf("插入后: ");
    PrintList(L);                 // 期望: 1 2 99 3 88 4 5

    // ========== 7. 删除: 删除第2个元素(值为2) ==========
    printf("\n========== 7. 删除第2个元素 ==========\n");
    ListDelete(L, 1, &e);
    printf("删除的值: %d, 删除后: ", e);
    PrintList(L);

    // ========== 8. 反向遍历(验证prior) ==========
    printf("\n========== 8. 反向遍历(验证prior) ==========\n");
    {
        LNode* p = L->prior;           // 尾结点
        printf("反向: ");
        while(p != L){
            printf("%d ", p->data);
            p = p->prior;
        }
        printf("\n");
    }

    // ========== 9. 删除尾结点 ==========
    printf("\n========== 9. 删除尾结点 ==========\n");
    int tailIdx = Length(L) - 1;
    ListDelete(L, tailIdx, &e);
    printf("删除尾结点值: %d, 删除后: ", e);
    PrintList(L);

    // ========== 10. 删除首结点 ==========
    printf("\n========== 10. 删除首结点 ==========\n");
    ListDelete(L, 0, &e);
    printf("删除首结点值: %d, 删除后: ", e);
    PrintList(L);

    // ========== 11. 销毁 ==========
    printf("\n========== 11. 销毁链表 ==========\n");
    DestroyList(L);
    printf("销毁完成\n");

    return 0;
}
