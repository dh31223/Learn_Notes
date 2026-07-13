#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

#define ElemType int


typedef struct SeqList{
    ElemType* data;
    int length;
    int MaxSize;
} SeqList;


// 第一题
ElemType func1(SeqList* S){
    if(S == NULL || S->length == 0){
        printf("列表为空！");
        return -1;
    }
    ElemType MinValue = __INT_MAX__;
    int MinValueIndex = 0;
    for(int i = 0; i < S->length; i++){
        if(S->data[i] < MinValue){
            MinValueIndex = i;
            MinValue = S->data[i];
        }
    }
    S->data[MinValueIndex] = S->data[S->length - 1];
    S->length -= 1;
    return MinValue;
}

// 第二题
bool func2(SeqList* S){
    if(S == NULL || S->length == 0)return false;
    int head = 0;
    int tail = S->length - 1;
    while(head < tail){
        ElemType temp = S->data[head];
        S->data[head] = S->data[tail];
        S->data[tail] = temp;
        head += 1;
        tail -= 1;
    }
    return true;
}

bool  func3(SeqList* S, ElemType x){
    if(S == NULL || S->length == 0)return false;
    int k = 0;
    for(int i = 0; i < S->length; i++){
        if(S->data[i] != x){
            S->data[k] = S->data[i];
            k++;
        }
    }
    S->length = k;
    return true;
}

bool func4(SeqList* S, ElemType s, ElemType t){
    if(S == NULL || S->length == 0 || s >= t)return false;
    int k = 0;
    for(int i = 0; i < S->length; i++){
        if(S->data[i] < s || S->data[i] > t){
            S->data[k] = S->data[i];
            k++;
        }
    }
    S->length = k;
    return true;
}


bool func5(SeqList* S){
    if(S == NULL || S->length == 0)return false;
    int k = 1;
    for(int i = 1; i < S->length; i++){
        if(S->data[i] != S->data[k - 1]){
            S->data[k++] = S->data[i];
        }
    }
    S->length = k;
    return true;
}


SeqList* func6(SeqList s1, SeqList s2){
    SeqList* temp = (SeqList*)malloc(sizeof(SeqList));
    temp->data = (ElemType*)malloc(sizeof(ElemType) * (s1.length + s2.length));
    temp->MaxSize = s1.length + s2.length;

    int Index = 0, i = 0, j = 0;
    // 主归并:两表都有剩余时,取较小的放进去
    while(i < s1.length && j < s2.length){
        if(s1.data[i] <= s2.data[j])
            temp->data[Index++] = s1.data[i++];
        else
            temp->data[Index++] = s2.data[j++];
    }
    // 收尾:把还有剩的那个表整段接上(两个 while 只会走一个)
    while(i < s1.length) temp->data[Index++] = s1.data[i++];
    while(j < s2.length) temp->data[Index++] = s2.data[j++];

    temp->length = Index;
    return temp;

}

// func7
void reverse(ElemType* data, int left, int right, int maxsize){
    if(left >= right || right - left > maxsize)return;
    int head = left, tail = right;
    while(head < tail){
        ElemType temp = data[head];
        data[head] = data[tail];
        data[tail] = temp;
        head += 1;
        tail -= 1;
    }
    return;
}

void exchange(ElemType* data, int m, int n){
    reverse(data, 0, m + n - 1, m + n - 1);
    reverse(data, 0, n - 1, m + n - 1);
    reverse(data, n, m + n - 1, m + n - 1);
    return;
}



// ===== 测试辅助函数 =====
// 用数组造一个顺序表
SeqList makeList(ElemType arr[], int n){
    SeqList L;
    L.data = (ElemType*)malloc(sizeof(ElemType) * (n > 0 ? n : 1));
    for(int i = 0; i < n; i++) L.data[i] = arr[i];
    L.length = n;
    L.MaxSize = n;
    return L;
}

// 打印顺序表
void printList(SeqList L){
    printf("[");
    for(int i = 0; i < L.length; i++){
        printf("%d", L.data[i]);
        if(i < L.length - 1) printf(", ");
    }
    printf("] (length=%d)", L.length);
}


int main(){
    printf("========== func1: 删除最小值元素,末元素填补空位 ==========\n");
    {
        ElemType arr[] = {3, 5, 1, 4, 2};   // 最小值 1(唯一)
        SeqList L = makeList(arr, 5);
        printf("原表:   "); printList(L); printf("\n");
        ElemType min = func1(&L);
        printf("被删最小值: %d\n", min);
        printf("删除后: "); printList(L); printf("\n");
        free(L.data);
    }

    printf("\n========== func2: 就地逆置 (O(1) 空间) ==========\n");
    {
        ElemType arr[] = {1, 2, 3, 4, 5};
        SeqList L = makeList(arr, 5);
        printf("原表:   "); printList(L); printf("\n");
        func2(&L);
        printf("逆置后: "); printList(L); printf("   (期望 5 4 3 2 1)\n");
        free(L.data);
    }

    printf("\n========== func3: 删除所有值为 x 的元素 (x=2) ==========\n");
    {
        ElemType arr[] = {1, 2, 2, 3, 2, 4};   // 含连续的 2,检验不漏删
        SeqList L = makeList(arr, 6);
        printf("原表:   "); printList(L); printf("\n");
        func3(&L, 2);
        printf("删除后: "); printList(L); printf("   (期望 1 3 4)\n");
        free(L.data);
    }

    printf("\n========== func4: 删除值在 [s,t] 内的元素 (s=5,t=7) ==========\n");
    {
        ElemType arr[] = {1, 5, 6, 7, 10, 2};
        SeqList L = makeList(arr, 6);
        printf("原表:   "); printList(L); printf("\n");
        func4(&L, 5, 7);
        printf("删除后: "); printList(L); printf("   (期望 1 10 2)\n");
        free(L.data);
    }

    printf("\n========== func5: 有序表去重 ==========\n");
    {
        ElemType arr[] = {1, 1, 2, 3, 3, 3, 5};
        SeqList L = makeList(arr, 7);
        printf("原表:   "); printList(L); printf("\n");
        func5(&L);
        printf("去重后: "); printList(L); printf("   (期望 1 2 3 5)\n");
        free(L.data);
    }

    printf("\n========== func6: 归并两个有序表为新表 ==========\n");
    {
        ElemType a1[] = {1, 3, 5, 7};
        ElemType a2[] = {2, 4, 6};
        SeqList s1 = makeList(a1, 4);
        SeqList s2 = makeList(a2, 3);
        printf("表1:    "); printList(s1); printf("\n");
        printf("表2:    "); printList(s2); printf("\n");
        SeqList* merged = func6(s1, s2);
        printf("归并后: "); printList(*merged); printf("   (期望 1..7)\n");
        free(s1.data);
        free(s2.data);
        free(merged->data);   // 先释放内部数组
        free(merged);         // 再释放结构体本身
    }

    printf("\n========== func7: 前 m 段与后 n 段互换 (m=3, n=2) ==========\n");
    {
        ElemType arr[] = {1, 2, 3, 4, 5};   // 前3个(1,2,3) 与 后2个(4,5) 互换
        SeqList L = makeList(arr, 5);
        printf("原表:   "); printList(L); printf("\n");
        exchange(L.data, 3, 2);
        printf("互换后: "); printList(L); printf("   (期望 4 5 1 2 3)\n");
        free(L.data);
    }

    return 0;
}