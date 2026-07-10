
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType int

typedef struct LNode{
    ElemType data;
    struct LNode* next;
}LNode, *LinkList;


// 带头结点的链表初始化

bool InitHeadList(LinkList* L){
    *L = (LNode* )malloc(sizeof(LNode));
    (*L)->next = NULL;
    return true;
}

// 不带头结点的链表初始化

bool InitList(LinkList* L){
    *L = NULL;
    return true;
}

// 带头结点
int Length(LinkList L){
    int len = 0;
    LNode* p = L;
    while(p->next != NULL){
        len++;
        p = p->next;
    }
    return len;
}

// 带头结点
LNode* GetElem(LinkList L, int Index){
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

// 按值查找节点

LNode* LocateElem(LinkList L, ElemType e){
    LNode* p = L->next;
    while(p != NULL && p->data != e){
        p = p->next;
    }
    return p;
}

// 后插
bool ListInsertBack(LinkList L, int Index, ElemType e){
    int len = Length(L);
    if(Index >= len || Index < 0){
        return false;
    }
    LNode* p = L;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    LNode* temp = (LNode* )malloc(sizeof(LNode));
    temp->data = e;
    temp->next = p->next;
    p->next = temp;
    return true;
}

// 前插
bool ListInsertFront(LinkList L, int Index, ElemType e){
    int len = Length(L);
    if(Index >= len || Index < 0){
        return false;
    }
    LNode* p = L;
    for(int i = 0; i < Index; i++){
        p = p->next;
    }
    LNode* temp = (LNode* )malloc(sizeof(LNode));
    temp->data = e;
    temp->next = p->next;
    p->next = temp;
    return true;
}

// 删除节点操作
bool ListDelete(LinkList L, int Index, ElemType* e){
    int len = Length(L);
    if(Index >= len || Index < 0){
        return false;
    }
    LNode* p = L;
    for(int i = 0; i < Index; i++){
        p = p->next;
    }
    LNode* temp = p->next;
    *e = p->next->data;
    p->next = p->next->next;
    free(temp);
    temp = NULL;
    return true;
}

// 头插法建立单链表
LinkList List_HeadInsert(LinkList L){
    LNode* temp; int x;
    L = (LNode* )malloc(sizeof(LNode));
    L->next = NULL;
    scanf("%d", &x);
    while(x != 9999){ // x == 9999 表述输入结束
        temp = (LNode* )malloc(sizeof(LNode));
        temp->data = x;
        temp->next = L->next;
        L->next = temp;
        scanf("%d", &x);
    }
    return L;
}

// 尾插法建立单链表
LinkList List_TailInsert(LinkList L){
    int x;
    L = (LNode* )malloc(sizeof(LNode));
    LNode* temp;
    LNode* r = L;
    scanf("%d", &x);
    while(x != 9999){
        temp = (LNode* )malloc(sizeof(LNode));
        temp->data = x;
        r->next = temp;
        r = temp;
        scanf("%d", &x);
    }
    r->next = NULL;
    return L;
}


// 打印链表
void PrintList(LinkList L){
    LNode* p = L->next;
    if(p == NULL){
        printf("(空链表)");
    }
    while(p != NULL){
        printf("%d", p->data);
        p = p->next;
        if(p != NULL) printf(" -> ");
    }
    printf("\n");
}

int main(){
    LinkList L;

    // ========== 1. 初始化 ==========
    printf("========== 1. 初始化带头结点的单链表 ==========\n");
    InitHeadList(&L);
    printf("长度: %d, 链表: ", Length(L));
    PrintList(L);

    // ========== 2. 构建初始链表 ==========
    // 注: 后插/前插要求 Index < len, 空表(len=0)无法调用
    // 这里手动插入第一个节点, 后续用后插构建
    printf("\n========== 2. 构建链表: 10 -> 20 -> 30 -> 40 -> 50 ==========\n");
    LNode* first = (LNode*)malloc(sizeof(LNode));
    first->data = 10; first->next = NULL;
    L->next = first;
    ListInsertBack(L, 0, 20);   // 10 后面插 20
    ListInsertBack(L, 1, 30);   // 20 后面插 30
    ListInsertBack(L, 2, 40);   // 30 后面插 40
    ListInsertBack(L, 3, 50);   // 40 后面插 50
    printf("链表: "); PrintList(L);
    printf("长度: %d\n", Length(L));

    // ========== 3. 后插测试 ==========
    printf("\n========== 3. 后插: 索引1(值为20)后插入 25 ==========\n");
    ListInsertBack(L, 1, 25);
    printf("链表: "); PrintList(L);
    printf("后插: 索引%d(尾节点)后插入 60 ==========\n", Length(L) - 1);
    ListInsertBack(L, Length(L) - 1, 60);
    printf("链表: "); PrintList(L);

    // ========== 4. 前插测试 ==========
    printf("\n========== 4. 前插测试 ==========\n");
    printf("索引0前插入 5: ");
    ListInsertFront(L, 0, 5);
    PrintList(L);
    printf("索引3前插入 18: ");
    ListInsertFront(L, 3, 18);
    PrintList(L);

    // ========== 5. 按位查找 ==========
    printf("\n========== 5. 按位查找 GetElem ==========\n");
    LNode* node = GetElem(L, 2);
    printf("索引2的元素: %d\n", node ? node->data : -1);
    node = GetElem(L, 0);
    printf("索引0的元素: %d\n", node ? node->data : -1);
    node = GetElem(L, 999);
    printf("索引999: %s\n", node ? "???" : "NULL (越界,正确)");

    // ========== 6. 按值查找 ==========
    printf("\n========== 6. 按值查找 LocateElem ==========\n");
    node = LocateElem(L, 25);
    printf("查找 25: %s\n", node ? "找到" : "未找到");
    node = LocateElem(L, 999);
    printf("查找 999: %s\n", node ? "???" : "未找到 (正确)");

    // ========== 7. 删除测试 ==========
    printf("\n========== 7. 删除测试 ==========\n");
    ElemType deleted;
    printf("删除前: "); PrintList(L);
    ListDelete(L, 2, &deleted);
    printf("删除索引2, 值为 %d: ", deleted); PrintList(L);
    ListDelete(L, 0, &deleted);
    printf("删除索引0(首节点), 值为 %d: ", deleted); PrintList(L);
    printf("删除后长度: %d\n", Length(L));

    // ========== 8. 边界测试 ==========
    printf("\n========== 8. 边界测试 ==========\n");
    printf("越界后插: %s\n", ListInsertBack(L, 999, 100) ? "成功???" : "失败 (正确)");
    printf("越界前插: %s\n", ListInsertFront(L, 999, 100) ? "成功???" : "失败 (正确)");
    printf("越界删除: %s\n", ListDelete(L, 999, &deleted) ? "成功???" : "失败 (正确)");
    printf("负数索引: %s\n", ListInsertBack(L, -1, 100) ? "成功???" : "失败 (正确)");

    // ========== 9. 头插法/尾插法(scanf, 需手动测试) ==========
    printf("\n========== 9. 头插法/尾插法 ==========\n");
    printf("这两个函数使用 scanf 输入, 输入 9999 结束.\n");
    printf("如需测试, 请取消下方注释:\n");
    // printf("--- 头插法, 输入数据(9999结束): ");
    // LinkList L2; L2 = List_HeadInsert(L2); PrintList(L2);
    // printf("--- 尾插法, 输入数据(9999结束): ");
    // LinkList L3; L3 = List_TailInsert(L3); PrintList(L3);

    printf("\n========== 全部测试完成 ==========\n");
    return 0;
}



