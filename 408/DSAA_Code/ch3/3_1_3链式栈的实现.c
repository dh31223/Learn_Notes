#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType int

typedef struct LinkStackNode{
    ElemType data;
    struct LinkStackNode* next;
}LinkStackNode, *LinkStackHead;


bool InitLinkStack(LinkStackHead* SH){
    if(*SH = (LinkStackNode* )malloc(sizeof(LinkStackNode))){
        (*SH)->next = NULL;
        return true;
    }
    return false;
}

bool StackEmpty(LinkStackHead SH){
    if(SH == NULL)return false;
    if(SH->next == NULL)return true;
    else return false;
}


bool Push(LinkStackHead SH, ElemType x){
    if(SH == NULL)return false;
    LinkStackNode* temp = (LinkStackNode* )malloc(sizeof(LinkStackNode));
    temp->data = x;
    temp->next = SH->next;
    SH->next = temp;
    return true;
}

bool Pop(LinkStackHead SH, ElemType* x){
    if(SH == NULL || SH->next == NULL)return false;
    *x = SH->next->data;
    LinkStackNode* temp = SH->next;
    SH->next = SH->next->next;
    free(temp);
    temp = NULL;
    return true;
}

bool GetTop(LinkStackHead SH, ElemType* x){
    if(SH == NULL || SH->next == NULL)return false;
    *x = SH->next->data;
    return true;
}

bool DestroyStack(LinkStackHead* SH){
    if(SH == NULL || *SH == NULL)return false;
    LinkStackNode* temp1 = (*SH)->next;
    while(temp1 != NULL){
        LinkStackNode* temp2 = temp1;
        temp1 = temp1->next;
        free(temp2);
    }
    free(*SH);
    (*SH) = NULL;
    return true;
}


int main(){
    LinkStackHead S;
    ElemType x;

    // 1. 初始化
    if(InitLinkStack(&S)) printf("[Init]  初始化成功\n");
    printf("[Empty] 初始化后是否为空: %s\n", StackEmpty(S) ? "空" : "非空");

    // 2. 依次入栈 1 2 3 4 5
    printf("\n[Push]  入栈: ");
    for(int i = 1; i <= 5; i++){
        Push(S, i);
        printf("%d ", i);
    }
    printf("\n[Empty] 入栈后是否为空: %s\n", StackEmpty(S) ? "空" : "非空");

    // 3. 读取栈顶（不出栈）
    if(GetTop(S, &x)) printf("[Top]   当前栈顶: %d\n", x);

    // 4. 全部出栈（应为 5 4 3 2 1，后进先出）
    printf("\n[Pop]   出栈: ");
    while(Pop(S, &x)){
        printf("%d ", x);
    }
    printf("\n[Empty] 出栈后是否为空: %s\n", StackEmpty(S) ? "空" : "非空");

    // 5. 边界：空栈继续 Pop / GetTop 应失败（链栈无上限，只测下溢）
    printf("\n[边界]  空栈 Pop:    %s\n", Pop(S, &x) ? "成功" : "失败(正确)");
    printf("[边界]  空栈 GetTop: %s\n", GetTop(S, &x) ? "成功" : "失败(正确)");

    // 6. 销毁，并验证调用者指针 S 被置空（二级指针的效果）
    if(DestroyStack(&S)) printf("\n[Destroy] 销毁成功\n");
    printf("[Destroy] 销毁后 S 是否为 NULL: %s\n", S == NULL ? "是(正确)" : "否");

    return 0;
}

