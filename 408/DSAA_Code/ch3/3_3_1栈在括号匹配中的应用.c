#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType char
#define MaxSize 50

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



bool DestroyStack(SqStack* S){
    if(S == NULL)return false;
    free(S->data);
    S->data = NULL;
    S->top = -1;
    return true;
}
int main(){
    SqStack S;
    int len;
    InitStack(&S);
    char str[50] = "";
    printf("请输入字符串长度：");
    scanf("%d", &len);
    if(len < 0)return 1;
    printf("请输入该字符串：");
    scanf("%s", str);
    for(int i = 0;i < len; i++){
        if(str[i] != '(' && str[i] != ')' && str[i] != '[' && str[i] != ']')continue;
        if(str[i] == '[' || str[i] == '('){
            Push(&S, str[i]);
        }
        if(str[i] == ')' || str[i] == ']'){
            char x;
            if(Pop(&S, &x)){
                if((x == '(' && str[i] != ')') || (x == '[' && str[i] != ']')){
                    printf("输入字符串错误！");
                    return 0;
                }
                else{
                    continue;
                }
            }
            else{
                printf("匹配失败！");
                DestroyStack(&S);
                return 0;
            }
        }
    }
    if(StackEmpty(&S)){
        printf("匹配完毕！");
    }
    else{
        printf("匹配失败！");
    }
    DestroyStack(&S);
    return 0;
}


