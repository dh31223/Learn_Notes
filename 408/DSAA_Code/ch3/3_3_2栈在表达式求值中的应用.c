#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

#define ElemType char
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

// 返回运算符优先级: * / 为2, + - 为1, ( 为0
int precedence(char op){
    if(op == '*' || op == '/') return 2;
    if(op == '+' || op == '-') return 1;
    return 0;  // '(' 在栈内优先级最低
}

int main(){
    char str[50];
    char ans[50];
    SqStack S;               // 修复1: 栈上分配，避免野指针
    int len;
    int index = 0;
    InitStack(&S);

    printf("请输入该表达式的长度：");
    scanf("%d", &len);
    printf("\n");
    printf("请输入一个表达式：");
    scanf("%s", str);
    printf("\n");

    for(int i = 0; i < len; i++){
        char c = str[i];

        // 判断是否为操作数(字母/数字)
        if((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')){
            ans[index++] = c;                     // 操作数直接输出
        }
        else if(c == '('){
            Push(&S, c);                          // 左括号直接入栈
        }
        else if(c == ')'){
            char x;
            while(Pop(&S, &x)){
                if(x == '(') break;               // 遇到左括号停止，丢弃它
                ans[index++] = x;                 // 弹出运算符输出
            }
        }
        else if(c == '+' || c == '-' || c == '*' || c == '/'){
            // 弹出栈中优先级 >= 当前运算符的所有运算符
            char top;
            while(GetTop(&S, &top) && top != '(' && precedence(top) >= precedence(c)){
                Pop(&S, &top);                    // 确认弹出
                ans[index++] = top;               // 输出
            }
            Push(&S, c);                          // 当前运算符入栈
        }
        // 其他字符(如空格)忽略
    }

    // 修复2: 将栈中剩余运算符全部弹出
    char x;
    while(Pop(&S, &x)){
        ans[index++] = x;
    }

    ans[index] = '\0';                            // 修复3: 添加字符串结尾
    printf("后缀表达式: %s\n", ans);

    DestroyStack(&S);                             // 修复4: 释放内存
    return 0;
}
