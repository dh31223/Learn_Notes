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


bool func4(LinkList L, ElemType start, ElemType end){
    if(L == NULL) return false;
    if(L->next == NULL || start > end) return false;
    LNode* temp = L;
    while(temp->next != NULL){
        if(temp->next->data > end || temp->next->data < start){
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

bool func5(LinkList L1, LinkList L2, LinkList* L){
    if(L1 == NULL || L2 == NULL || L == NULL || *L == NULL)return false;
    if(L1->next == NULL || L2->next == NULL)return false;
    LNode* temp1 = L1;
    LNode* temp2 = L2;
    LNode* temp3 = (*L);
    while(temp1->next != NULL){
        temp1 = temp1->next;
        temp2 = L2;
        while(temp2->next != NULL){
            temp2 = temp2->next;
            if(temp1->data != temp2->data)continue;
            LNode* p = (LNode* )malloc(sizeof(LNode));
            p->data = temp2->data;
            p->next = NULL;
            temp3->next = p;
            temp3 = temp3->next;
            break;
        }
    }
    return true;
}

bool func6(LinkList L, LinkList* LA, LinkList* LB){
    if(L == NULL || LA == NULL || LB == NULL || (*LA) == NULL || (*LB) == NULL)return false;
    if(L->next == NULL)return false;
    LNode* A = L->next;
    LNode* B = L->next->next;
    LNode* tempA = L->next;
    LNode* tempB = L->next->next;
    if(L->next->next->next != NULL){
        LNode* tempNow = L->next;
        while(tempNow->next->next != NULL){
            tempNow = tempNow->next->next;
            tempA->next = tempNow;
            tempB->next = tempNow->next;
            tempA = tempA->next;
            tempB = tempB->next;
        }
    }
    tempA->next = NULL;
    tempB->next = NULL;
    (*LA)->next = A;
    (*LB)->next = B;
    return true;
}

bool func7(LinkList* L){
    if(L == NULL || (*L) == NULL)return false;
    if((*L)->next == NULL)return false;
    LNode* temp = (*L)->next;
    while(temp != NULL && temp->next != NULL){
        if(temp->data != temp->next->data){
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



int main(){
    return 0;
}


