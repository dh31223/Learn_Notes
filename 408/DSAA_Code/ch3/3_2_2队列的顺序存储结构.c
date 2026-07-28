#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType int
#define MaxSize 10

typedef struct SqQueue{
    ElemType* data;
    int front, rear;
}SqQueue;

bool InitQueue(SqQueue* Q){
    if(Q == NULL)return false;
    Q->data = (ElemType* )malloc(sizeof(ElemType) * MaxSize);
    Q->front = 0;
    Q->rear = 0;
    return true;
}

bool QueueEmpty(SqQueue Q){
    return Q.front == Q.rear;
}

bool EnQueue(SqQueue* Q, ElemType x){
    if(Q == NULL || Q->rear == MaxSize)return false;
    Q->data[Q->rear++] = x;
    return true;
}

bool DeQueue(SqQueue* Q, ElemType* x){
    if(Q == NULL || Q->front == Q->rear)return false;
    *x = Q->data[Q->front++];
    return true;
}

bool GetHead(SqQueue Q, ElemType* x){
    if(Q.front == Q.rear)return false;
    *x = Q.data[Q.front];
    return true;
}

bool GetTail(SqQueue Q, ElemType* x){
    if(Q.front == Q.rear)return false;
    *x = Q.data[Q.rear - 1];
    return true;

}

bool DestroyQueue(SqQueue* Q){
    if(Q == NULL)return false;
    free(Q->data);
    Q->data = NULL;
    return true;
}


void printQueue(SqQueue Q){
    printf("队列(队头→队尾): [");
    for(int i = Q.front; i < Q.rear; i++){
        printf("%d", Q.data[i]);
        if(i < Q.rear - 1) printf(", ");
    }
    printf("]  (front=%d, rear=%d)", Q.front, Q.rear);
}

int main(){
    SqQueue Q;
    ElemType x;

    printf("========== 初始化 ==========\n");
    InitQueue(&Q);
    printf("初始化后是否为空: %s\n", QueueEmpty(Q) ? "空" : "非空");

    printf("\n========== 入队 1 2 3 4 5 ==========\n");
    for(int i = 1; i <= 5; i++){
        EnQueue(&Q, i);
        printf("入队 %d  ", i);
    }
    printf("\n"); printQueue(Q); printf("\n");
    printf("是否为空: %s\n", QueueEmpty(Q) ? "空" : "非空");

    printf("\n========== 查看队头 / 队尾 ==========\n");
    GetHead(Q, &x); printf("队头元素: %d  (期望 1)\n", x);
    GetTail(Q, &x); printf("队尾元素: %d  (期望 5)\n", x);

    printf("\n========== 出队(应为 1 2 3 4 5,先进先出) ==========\n");
    printf("出队顺序: ");
    while(DeQueue(&Q, &x)) printf("%d ", x);
    printf("\n是否为空: %s\n", QueueEmpty(Q) ? "空" : "非空");

    printf("\n========== 边界:空队 出队 / 取队头 / 取队尾 ==========\n");
    printf("空队 DeQueue: %s\n", DeQueue(&Q, &x) ? "成功" : "失败(正确)");
    printf("空队 GetHead: %s\n", GetHead(Q, &x) ? "成功" : "失败(正确)");
    printf("空队 GetTail: %s\n", GetTail(Q, &x) ? "成功" : "失败(正确)");

    printf("\n========== 边界:入队到满(MaxSize = %d) ==========\n", MaxSize);
    DestroyQueue(&Q);           // 清掉旧的,重新初始化,便于干净演示
    InitQueue(&Q);
    int cnt = 0;
    while(EnQueue(&Q, cnt)) cnt++;
    printf("连续入队,成功 %d 个后队满(rear = %d)\n", cnt, Q.rear);
    printf("队满再入队 999: %s\n", EnQueue(&Q, 999) ? "成功" : "失败(正确)");

    printf("\n========== 演示线性队列的“假溢出” ==========\n");
    DeQueue(&Q, &x); DeQueue(&Q, &x);   // 出队 2 个,前面腾出 2 个空位
    printf("出队 2 个后:front = %d, rear = %d,前面已空出 2 个位置\n", Q.front, Q.rear);
    printf("此时再入队: %s  ← 明明前面有空位,却因 rear==MaxSize 被判满\n",
           EnQueue(&Q, 888) ? "成功" : "失败(假溢出!)");
    printf("这就是“假溢出”,正是循环队列要解决的问题。\n");

    DestroyQueue(&Q);
    return 0;
}
