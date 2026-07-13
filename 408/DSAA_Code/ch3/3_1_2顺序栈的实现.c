#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType int
#define MaxSize 30

typedef struct SqStack{
    ElemType* data;
    int top;
} SqStack;

bool InitStack(SqStack* S){
    if(S == NULL)return false;
    S->data = (ElemType* )malloc(sizeof(ElemType) * MaxSize);
    S->top = -1;
    return true;
}

bool StackEmpty(SqStack* S){
    if(S->top == -1)return true;
    else return false;
}

bool Push(SqStack* S, ElemType x){
    if(S->top + 1 == MaxSize)return false;
    S->data[S->top + 1] = x;
    S->top += 1;
    return true;
}

bool Pop(SqStack* S, ElemType* x){
    if(S->top == -1)return false;
    *x = S->data[S->top];
    S->top -= 1;
    return true;
}

bool GetTop(SqStack* S, ElemType* x){
    if(S->top == -1)return false;
    *x = S->data[S->top];
    return true;
}

bool DestroyStack(SqStack* S){
    if(S == NULL)return false;
    free(S->data);
    S->data = NULL;
    S->top = -1;
    return true;
}


int main(){
    SqStack s;
    ElemType x;

    // 1. 初始化
    if(InitStack(&s)) printf("[Init]  初始化成功\n");
    printf("[Empty] 初始化后是否为空: %s\n", StackEmpty(&s) ? "空" : "非空");

    // 2. 依次入栈 1 2 3 4 5
    printf("\n[Push]  入栈: ");
    for(int i = 1; i <= 5; i++){
        Push(&s, i);
        printf("%d ", i);
    }
    printf("\n[Empty] 入栈后是否为空: %s\n", StackEmpty(&s) ? "空" : "非空");

    // 3. 读取栈顶（不出栈）
    if(GetTop(&s, &x)) printf("[Top]   当前栈顶: %d\n", x);

    // 4. 全部出栈（应为 5 4 3 2 1，后进先出）
    printf("\n[Pop]   出栈: ");
    while(Pop(&s, &x)){
        printf("%d ", x);
    }
    printf("\n[Empty] 出栈后是否为空: %s\n", StackEmpty(&s) ? "空" : "非空");

    // 5. 边界：空栈继续 Pop / GetTop 应失败
    printf("\n[边界]  空栈 Pop:    %s\n", Pop(&s, &x) ? "成功" : "失败(正确)");
    printf("[边界]  空栈 GetTop: %s\n", GetTop(&s, &x) ? "成功" : "失败(正确)");

    // 6. 边界：压满 MaxSize 个后再入栈应失败（栈满）
    for(int i = 0; i < MaxSize; i++) Push(&s, i);
    printf("[边界]  压满后再 Push: %s\n", Push(&s, 999) ? "成功" : "失败(正确)");

    // 7. 销毁
    if(DestroyStack(&s)) printf("\n[Destroy] 销毁成功\n");

    return 0;
}