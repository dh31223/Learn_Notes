
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType int

typedef struct DLNode{
    ElemType data;
    struct DLNode* prior;
    struct DLNode* next;
}DLNode, *DLinkList;


bool InitDList(DLinkList* L){
    *L = (DLNode* )malloc(sizeof(DLNode));
    (*L)->next = NULL;
    (*L)->prior = NULL;
    return true;
}

// 获取长度
int Length(DLinkList L){
    DLNode* p = L;
    int len = 0;
    while(p->next != NULL){
        p = p->next;
        len++;
    }
    return len;
}


DLNode* GetElem(DLinkList L, int Index){
    int len = Length(L);
    if(Index >= len || Index < 0){
        return NULL;
    }
    DLNode* p = L;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    return p;
}


DLNode* LocateElem(DLinkList L, ElemType e){
    DLNode* p = L->next;
    while(p != NULL && p->data != e){
        p = p->next;
    }
    return p;
}


// 后插
bool ListInsertBack(DLinkList L, int Index, ElemType e){
    int len = Length(L);
    if(Index >= len || Index < 0){
        return false;
    }
    DLNode* p = L;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    DLNode* temp = (DLNode* )malloc(sizeof(DLNode));
    temp->data = e;
    temp->prior = p;
    temp->next = p->next;
    if(p->next != NULL){
        p->next->prior = temp;
    }
    p->next = temp;
    return true;
}

// 前插
bool ListInsertFront(DLinkList L, int Index, ElemType e){
    int len = Length(L);
    if(Index >= len || Index < 0){
        return false;
    }
    DLNode* p = L;
    for(int i = 0; i < Index; i++){
        p = p->next;
    }
    DLNode* temp = (DLNode* )malloc(sizeof(DLNode));
    temp->data = e;
    temp->next = p->next;
    temp->prior = p;
    p->next->prior = temp;
    p->next = temp;
    return true;
}

bool ListDelete(DLinkList L, int Index, ElemType* e){
    int len = Length(L);
    if(Index >= len || Index < 0){
        return false;
    }
    DLNode* p = L;
    for(int i = 0; i <= Index; i++){
        p = p->next;
    }
    *e = p->data;
    p->prior->next = p->next;
    if(p->next != NULL){
        p->next->prior = p->prior;
    }
    free(p);
    p = NULL;
    return true;
}

// 头插法
DLinkList List_HeadInsert(DLinkList L){
    DLNode* temp;
    int x;
    scanf("%d", &x);
    while(x != 9999){
        temp = (DLNode* )malloc(sizeof(DLNode));
        temp->data = x;
        temp->next = L->next;
        temp->prior = L;
        if(L->next != NULL){
            L->next->prior = temp;
        }
        L->next = temp;
        scanf("%d", &x);
    }
    return L;
}


// 尾插法
DLinkList List_TailInsert(DLinkList L){
    int x;
    DLNode *temp, *tail;
    tail = L;
    while(tail->next != NULL){
        tail = tail->next;
    }
    scanf("%d", &x);
    while(x != 9999){
        temp = (DLNode* )malloc(sizeof(DLNode));
        temp->data = x;
        temp->next = tail->next;
        temp->prior = tail;
        tail->next = temp;
        tail = temp;
        scanf("%d", &x);
    }
    return L;
}

// 打印
void PrintList(DLinkList L){
    int len = Length(L);
    DLNode* p = L;
    for(int i = 0; i < len; i++){
        p = p->next;
        printf("%d ", p->data);
    }
    return;
}



int main(){
    DLinkList L;
    ElemType e;

    // ===== 1. 尾插法建表 =====
    printf("=== 尾插法建表 (输入: 1 2 3 9999) ===\n");
    InitDList(&L);
    List_TailInsert(L);
    printf("正向: "); PrintList(L); printf("\n");

    // ===== 2. 反向遍历 (验证prior) =====
    printf("反向: ");
    DLNode* p = L;
    while(p->next != NULL) p = p->next;   // 走到尾节点
    while(p != L){
        printf("%d ", p->data);
        p = p->prior;
    }
    printf("\n");

    // ===== 3. 基本操作测试 =====
    printf("\n=== 基本操作 ===\n");
    printf("长度: %d\n", Length(L));

    printf("按位查找[0]: %d\n", GetElem(L, 0)->data);
    printf("按位查找[2]: %d\n", GetElem(L, 2)->data);
    printf("按值查找(2): %s\n", LocateElem(L, 2) ? "找到" : "未找到");
    printf("按值查找(99): %s\n", LocateElem(L, 99) ? "找到" : "未找到");

    // ===== 4. 后插 & 前插 =====
    printf("\n=== 插入测试 ===\n");
    ListInsertBack(L, 0, 10);              // 第1个元素后插10
    printf("下标0后插10: "); PrintList(L); printf("\n");
    ListInsertFront(L, 0, 20);             // 第1个元素前插20
    printf("下标0前插20: "); PrintList(L); printf("\n");
    ListInsertBack(L, Length(L)-1, 99);    // 最后元素后插99
    printf("末尾后插99: "); PrintList(L); printf("\n");

    // 验证反向
    printf("反向验证: ");
    p = L; while(p->next != NULL) p = p->next;
    while(p != L){ printf("%d ", p->data); p = p->prior; }
    printf("\n");

    // ===== 5. 删除测试 =====
    printf("\n=== 删除测试 ===\n");
    ListDelete(L, 0, &e);
    printf("删除[0]=%d: ", e); PrintList(L); printf("\n");
    ListDelete(L, Length(L)-1, &e);
    printf("删除末尾=%d: ", e); PrintList(L); printf("\n");

    printf("反向验证: ");
    p = L; while(p->next != NULL) p = p->next;
    while(p != L){ printf("%d ", p->data); p = p->prior; }
    printf("\n");

    // ===== 6. 头插法建新表 =====
    printf("\n=== 头插法建新表 (输入: 1 2 3 9999) ===\n");
    DLinkList L2;
    InitDList(&L2);
    List_HeadInsert(L2);
    printf("正向(应为逆序 3 2 1): "); PrintList(L2); printf("\n");
    printf("反向: ");
    p = L2; while(p->next != NULL) p = p->next;
    while(p != L2){ printf("%d ", p->data); p = p->prior; }
    printf("\n");

    return 0;
}



