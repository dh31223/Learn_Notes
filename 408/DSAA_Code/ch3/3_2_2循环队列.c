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

bool EmptyQueue(SqQueue Q){
    return Q.front == Q.rear;
}

bool FullQueue(SqQueue Q){
    return (Q.rear + 1) % MaxSize == Q.front;
}

bool EnQueue(SqQueue* Q, ElemType x){
    if(Q == NULL)return false;
    if(FullQueue(*Q))return false;
    Q->data[Q->rear] = x;
    Q->rear = (Q->rear + 1) % MaxSize;
    return true;
}

bool DeQueue(SqQueue* Q, ElemType* x){
    if(Q == NULL)return false;
    if(EmptyQueue(*Q))return false;
    *x = Q->data[Q->front];
    Q->front = (Q->front + 1) % MaxSize;
    return true;
}


bool GetHead(SqQueue Q, ElemType* x){
    if(EmptyQueue(Q))return false;
    *x = Q.data[Q.front];
    return true;
}

bool DestroyQueue(SqQueue* Q){
    if(Q == NULL)return false;
    free(Q->data);
    Q->data = NULL;
    return true;
}

void PrintQueue(SqQueue Q){
    int Index = Q.front;
    while(Index != Q.rear){
        printf("%d ", Q.data[Index]);
        Index = (Index + 1) % MaxSize;
    }
    return;
}


int Length(SqQueue Q){
    return (Q.rear + MaxSize - Q.front) % MaxSize;
}

int main(){
    SqQueue Q;
    ElemType x;

    printf("========== 初始化 ==========\n");
    printf("InitQueue 返回: %s\n", InitQueue(&Q) ? "true(成功)" : "false");
    printf("是否为空: %s\n", EmptyQueue(Q) ? "空" : "非空");

    printf("\n========== 入队 1 2 3 4 5 ==========\n");
    for(int i = 1; i <= 5; i++) EnQueue(&Q, i);
    printf("队列内容(队头→队尾): "); PrintQueue(Q); printf("  (共 %d 个)\n", QueueLength(Q));
    GetHead(Q, &x); printf("队头元素: %d\n", x);

    printf("\n========== 出队(先进先出) ==========\n");
    printf("出队顺序: ");
    while(DeQueue(&Q, &x)) printf("%d ", x);
    printf("\n是否为空: %s\n", EmptyQueue(Q) ? "空" : "非空");

    printf("\n========== 边界:空队 出队 / 取队头 ==========\n");
    printf("空队 DeQueue: %s\n", DeQueue(&Q, &x) ? "成功" : "失败(正确)");
    printf("空队 GetHead: %s\n", GetHead(Q, &x) ? "成功" : "失败(正确)");

    printf("\n========== 边界:入队到满(循环队列牺牲一格,最多存 %d 个) ==========\n", MaxSize - 1);
    int cnt = 0;
    while(EnQueue(&Q, cnt)) cnt++;
    printf("连续入队,成功 %d 个后判满(front=%d, rear=%d)\n", cnt, Q.front, Q.rear);
    printf("此刻 队满=%s, 队空=%s  ← 满与空能正确区分\n",
           FullQueue(Q) ? "是" : "否", EmptyQueue(Q) ? "是" : "否");
    printf("队满再入队 999: %s\n", EnQueue(&Q, 999) ? "成功" : "失败(正确)");

    printf("\n========== 关键:验证绕回(不再假溢出) ==========\n");
    DeQueue(&Q, &x); printf("出队: %d\n", x);
    DeQueue(&Q, &x); printf("出队: %d\n", x);
    printf("出队 2 个后 front=%d, rear=%d\n", Q.front, Q.rear);
    printf("再入队 100: %s  ← 前面空位被绕回复用\n", EnQueue(&Q, 100) ? "成功" : "失败(假溢出)");
    printf("再入队 200: %s\n", EnQueue(&Q, 200) ? "成功" : "失败(假溢出)");
    printf("队列内容(队头→队尾): "); PrintQueue(Q);
    printf("  (共 %d 个, rear 已绕回到 %d)\n", QueueLength(Q), Q.rear);

    DestroyQueue(&Q);
    return 0;
}
