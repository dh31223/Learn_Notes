#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#define ElemType int

typedef struct LNode{
    ElemType data;
    struct LNode* next;
}LNode, *LinkList;


bool func1(LinkList L, ElemType x){
    if(L == NULL) return false;
    LNode* temp = L;
    while(temp->next != NULL){
        if(temp->next->data != x){
            temp = temp->next;
            continue;
        }
        LNode* p = temp->next;
        temp->next = temp->next->next;
        free(p);
        p = NULL;
    }
    return true;
}


bool func2(LinkList L){
    if(L == NULL)return false;
    if(L->next == NULL)return false;
    LNode* temp = L;
    LNode* MinNode = L;
    ElemType MinValue = temp->next->data;
    while(temp->next != NULL){
        if(temp->next->data >= MinValue){
            temp = temp->next;
            continue;
        }
        MinNode = temp;
        MinValue = temp->next->data;
        temp = temp->next;
    }
    temp = MinNode->next;
    MinNode->next = MinNode->next->next;
    free(temp);
    temp = NULL;
    return true;
}


bool func3(LinkList L){
    if(L == NULL)return false;
    if(L->next == NULL)return false;
    LNode* temp = L->next;
    L->next = NULL;
    while(temp != NULL){
        LNode* p = temp->next;
        temp->next = L->next;
        L->next = temp;
        temp = p;
    }
    return true;
}

int main(){
    return 0;
}


