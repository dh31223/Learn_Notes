
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

int StrLength(const char* S){
    int len = 0;
    while(S[len] != '\0') len++;
    return len;
}

bool StrEmpty(const char* S){
    return S[0] == '\0';
}

int StrCompare(const char* S, const char* T){
    int index = 0;
    while(S[index] != '\0' && T[index] != '\0'){
        if(S[index] != T[index]){
            return S[index] - T[index];
        }
        index++;
    }
    return S[index] - T[index];
}

char* StrAssign(const char* chars){
    int len = StrLength(chars);
    char* S = (char* )malloc((len + 1) * sizeof(char));
    for(int i = 0; i <= len; i++){
        S[i] = chars[i];
    }
    return S;
}

char* StrCopy(const char* S){
    return StrAssign(S);
}

char* SubString(const char* S, int pos, int len){
    int lenS = StrLength(S);
    // 边界检查: pos非法或len<=0时返回空串
    if(pos < 0 || pos >= lenS || len <= 0){
        char* Sub = (char* )malloc(sizeof(char));
        Sub[0] = '\0';
        return Sub;
    }
    if(pos + len > lenS){
        len = lenS - pos;
    }
    char* Sub = (char* )malloc((len + 1) * sizeof(char));
    for(int i = 0; i < len; i++){
        Sub[i] = S[pos + i];
    }
    Sub[len] = '\0';
    return Sub;
}

char* Concat(const char* s1, const char* s2){
    int len1 = StrLength(s1);
    int len2 = StrLength(s2);
    char* T = (char* )malloc((len1 + len2 + 1) * sizeof(char));
    for(int i = 0; i < len1; i++)T[i] = s1[i];
    for(int i = 0; i < len2; i++)T[len1 + i] = s2[i];
    T[len1 + len2] = '\0';
    return T;
}

int Index(const char* S, const char* T){
    int lenS = StrLength(S);
    int lenT = StrLength(T);
    if(lenT == 0)return 0;
    if(lenT > lenS)return -1;
    for (int i = 0; i <= lenS - lenT; i++) {
        int j;
        for (j = 0; j < lenT; j++) {
            if (S[i + j] != T[j])
                break;
        }
        if (j == lenT)  // 全部匹配成功
            return i;
    }
    return -1;
}


void ClearString(char *S) {
    S[0] = '\0';
}

void DestroyString(char **S) {
    free(*S);
    *S = NULL;
}

int main(){
    printf("========== 串基本操作测试 ==========\n\n");

    // ---------- 1. StrAssign & StrLength ----------
    printf("--- StrAssign & StrLength ---\n");
    char* s1 = StrAssign("Hello");
    char* s2 = StrAssign("HelloWorld");
    char* s3 = StrAssign("");
    printf("s1 = \"%s\",  StrLength = %d\n", s1, StrLength(s1));
    printf("s2 = \"%s\",  StrLength = %d\n", s2, StrLength(s2));
    printf("s3 = \"%s\",      StrLength = %d\n", s3, StrLength(s3));

    // ---------- 2. StrEmpty ----------
    printf("\n--- StrEmpty ---\n");
    printf("StrEmpty(\"%s\")  = %s\n", s1, StrEmpty(s1) ? "true" : "false");
    printf("StrEmpty(\"%s\")      = %s\n", s3, StrEmpty(s3) ? "true" : "false");

    // ---------- 3. StrCompare ----------
    printf("\n--- StrCompare ---\n");
    char* abc1 = StrAssign("abc");
    char* abc2 = StrAssign("abc");
    char* abd  = StrAssign("abd");
    char* ab   = StrAssign("ab");
    printf("StrCompare(\"abc\", \"abc\") = %d  (相等,应为0)\n", StrCompare(abc1, abc2));
    printf("StrCompare(\"abc\", \"abd\") = %d  (小,  应为负)\n", StrCompare(abc1, abd));
    printf("StrCompare(\"abd\", \"abc\") = %d  (大,  应为正)\n", StrCompare(abd, abc1));
    printf("StrCompare(\"abc\", \"ab\")  = %d  (长,  应为正)\n", StrCompare(abc1, ab));
    printf("StrCompare(\"ab\",  \"abc\") = %d  (短,  应为负)\n", StrCompare(ab, abc1));

    // ---------- 4. StrCopy ----------
    printf("\n--- StrCopy ---\n");
    char* copy = StrCopy(s1);
    printf("StrCopy(\"%s\") = \"%s\", 比较结果 = %d\n",
           s1, copy, StrCompare(s1, copy));

    // ---------- 5. SubString ----------
    printf("\n--- SubString ---\n");
    printf("SubString(\"%s\", 0, 5) = \"%s\"   (取\"Hello\")\n",
           s2, SubString(s2, 0, 5));
    printf("SubString(\"%s\", 5, 5) = \"%s\"   (取\"World\")\n",
           s2, SubString(s2, 5, 5));
    printf("SubString(\"%s\", 5, 100) = \"%s\" (超长自动截断)\n",
           s2, SubString(s2, 5, 100));
    // 边界测试: pos非法
    char* bad1 = SubString(s2, -1, 3);
    printf("SubString(\"%s\", -1, 3) = \"%s\"  (pos<0返回空串)\n", s2, bad1);
    char* bad2 = SubString(s2, 100, 3);
    printf("SubString(\"%s\", 100, 3)= \"%s\"  (pos越界返回空串)\n", s2, bad2);
    DestroyString(&bad1);
    DestroyString(&bad2);

    // ---------- 6. Concat ----------
    printf("\n--- Concat ---\n");
    char* hello = StrAssign("Hello");
    char* world = StrAssign("World");
    char* cat1  = Concat(hello, world);
    printf("Concat(\"Hello\", \"World\") = \"%s\"\n", cat1);
    char* empty = StrAssign("");
    char* cat2  = Concat(hello, empty);
    printf("Concat(\"Hello\", \"\")      = \"%s\"   (拼接空串)\n", cat2);
    char* cat3  = Concat(empty, world);
    printf("Concat(\"\", \"World\")      = \"%s\"   (空串拼接)\n", cat3);

    // ---------- 7. Index (核心!) ----------
    printf("\n--- Index (模式匹配) ---\n");
    printf("Index(\"%s\", \"%s\") = %2d  (\"World\"在位置5)\n",
           s2, world, Index(s2, world));
    printf("Index(\"%s\", \"%s\") = %2d  (\"Hello\"在位置0)\n",
           s2, hello, Index(s2, hello));
    printf("Index(\"%s\", \"%s\") = %2d  (不存在,应为-1)\n",
           s2, "xyz", Index(s2, "xyz"));
    printf("Index(\"%s\", \"%s\") = %2d  (空串是任何串的子串,位置0)\n",
           s2, "", Index(s2, ""));
    printf("Index(\"%s\", \"%s\") = %2d  (主串比模式串短,应为-1)\n",
           "ab", "abcd", Index("ab", "abcd"));
    // 重复匹配: 第一个"lo"在位置2
    printf("Index(\"%s\", \"%s\") = %2d  (找第一个\"lo\")\n",
           hello, "lo", Index(hello, "lo"));

    // ---------- 8. ClearString ----------
    printf("\n--- ClearString ---\n");
    printf("清空前 cat1 = \"%s\", StrLength = %d\n", cat1, StrLength(cat1));
    ClearString(cat1);
    printf("清空后 cat1 = \"%s\", StrLength = %d, StrEmpty = %s\n",
           cat1, StrLength(cat1), StrEmpty(cat1) ? "true" : "false");

    // ---------- 9. DestroyString ----------
    printf("\n--- DestroyString ---\n");
    printf("销毁前 s1 = \"%s\"\n", s1);
    DestroyString(&s1);
    printf("销毁后 s1 = %p  (应为nil/null)\n\n", (void*)s1);

    // ---------- 清理 ----------
    DestroyString(&s2);
    DestroyString(&s3);
    DestroyString(&copy);
    DestroyString(&abc1);
    DestroyString(&abc2);
    DestroyString(&abd);
    DestroyString(&ab);
    DestroyString(&hello);
    DestroyString(&world);
    DestroyString(&cat1);
    DestroyString(&cat2);
    DestroyString(&cat3);
    DestroyString(&empty);

    printf("========== 全部测试完成 ==========\n");
    return 0;
}

