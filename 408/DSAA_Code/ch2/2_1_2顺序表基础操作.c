
#include<stdio.h>
#include<stdbool.h>
#include<stdlib.h>
#define ElemType int

// 结构体
typedef struct SeqList{
    ElemType* data;
    int length;
    int MaxSize;
}SeqList;

// 初始化顺序表
void InitList(SeqList* L, int InitSize){
    L->data = (ElemType *)malloc(InitSize * sizeof(ElemType));
    L->length = 0;
    L->MaxSize = InitSize;
    return;
}

// 销毁顺序表
void DestroyList(SeqList* L){
    free(L->data);
    L->data = NULL;
    return;
}

// 插入
bool ListInsert(SeqList* L, int Index, ElemType e){
    if(Index > L->length || Index < 0){
        printf("插入位置不合理，插入失败！\n");
        return false;
    }

    if(L->length == L->MaxSize){
        printf("长度达到上限，插入失败！\n");
        return false;
    }

    for(int i = L->length - 1; i >= Index; i--){
        L->data[i + 1] = L->data[i];
    }
    L->data[Index] = e;
    L->length++;
    printf("插入成功！\n");
    return true;
}

// 删除
bool ListDelete(SeqList* L, int Index, ElemType* e){
    if(Index < 0 || Index >= L->length){
        printf("删除位置不合理，删除失败！\n");
        return false;
    }

    *e = L->data[Index];
    L->length--;

    for(int i = Index; i < L->length; i++){
        L->data[i] = L->data[i + 1];
    }

    printf("删除成功！\n");
    return true;
}

// 根据元素查找下标
int LocateElem(SeqList L, ElemType e){
    for(int i = 0; i < L.length; i++){
        if(L.data[i] == e){
            printf("查找成功下标为%d\n", i);
            return i;
        }
    }
    printf("查找失败，不存在该元素！\n");
    return -1;
}

// 更具下标查找元素
ElemType GetElem(SeqList L, int Index){

    if(Index < 0 || Index >= L.MaxSize){
        printf("下标不合理，获取失败！\n");
        return -1;
    }
    return L.data[Index];
}

// 获取顺序表的长度
int Length(SeqList L){
    return L.length;
}

// 打印整个顺序表
void PrintList(SeqList L){

    if(L.length == 0){
        printf("该顺序表为空！\n");
        return;
    }

    for(int i = 0; i < L.length; i++){
        printf("%d ", L.data[i]);
    }
    return;
}

// 判断顺序表是否为空
bool Empty(SeqList L){
    if(L.length == 0){
        return true;
    }
    return false;
}

// 扩充顺序表
void IncreaseSize(SeqList* L, int Increase_Size){
    ElemType* new_data = (ElemType* )malloc(sizeof(ElemType) * (L->MaxSize + Increase_Size));
    for(int i = 0; i < L->length; i++){
        new_data[i] = L->data[i];
    }
    free(L->data);
    L->data = new_data;
    L->MaxSize += Increase_Size;
    return;
}


int main(){
    SeqList L;
    InitList(&L, 10);

    // 测试插入
    printf("=== 插入 5 个元素 ===\n");
    ListInsert(&L, 0, 10);
    ListInsert(&L, 1, 20);
    ListInsert(&L, 2, 30);
    ListInsert(&L, 1, 15);   // 中间插入
    ListInsert(&L, L.length, 40);  // 尾插
    printf("\n当前顺序表: ");
    PrintList(L);
    printf("长度: %d\n\n", Length(L));

    // 测试按值查找
    printf("=== 查找元素 15 ===\n");
    int pos = LocateElem(L, 15);
    printf("\n");
    printf("=== 查找元素 99 ===\n");
    LocateElem(L, 99);
    printf("\n\n");

    // 测试按位查找
    printf("=== 查找下标 2 的元素 ===\n");
    printf("%d\n\n", GetElem(L, 2));

    // 测试删除
    printf("=== 删除下标 1 的元素 ===\n");
    ElemType deleted;
    ListDelete(&L, 1, &deleted);
    printf("\n删除的元素值: %d\n", deleted);
    printf("删除后: ");
    PrintList(L);
    printf("长度: %d\n\n", Length(L));

    // 测试扩容
    printf("=== 测试扩容 ===\n");
    printf("当前最大容量: %d, 当前长度: %d\n", L.MaxSize, L.length);
    // 先填满剩余空间（当前长度4，容量10，再插6个填满）
    for(int i = 0; i < 6; i++){
        ListInsert(&L, L.length, 100 + i);
    }
    printf("填满后: ");
    PrintList(L);
    printf("长度: %d, 最大容量: %d\n\n", L.length, L.MaxSize);

    // 满了之后再插入，应该失败
    printf("=== 满容量时插入 ===\n");
    ListInsert(&L, 0, 999);
    printf("\n");

    // 扩容 5
    printf("=== 调用 IncreaseSize 扩容 5 ===\n");
    IncreaseSize(&L, 5);
    printf("扩容后最大容量: %d\n\n", L.MaxSize);

    // 再插入，应该成功
    printf("=== 扩容后插入 ===\n");
    ListInsert(&L, 0, 999);
    printf("当前顺序表: ");
    PrintList(L);
    printf("长度: %d, 最大容量: %d\n\n", L.length, L.MaxSize);

    // 测试判空
    printf("=== 判空 ===\n");
    printf("顺序表%s\n\n", Empty(L) ? "为空" : "不为空");

    // 销毁
    DestroyList(&L);
    printf("=== 销毁后判空 ===\n");
    printf("顺序表%s\n", L.data == NULL ? "已销毁" : "未销毁");

    return 0;
}


