/*
 * 4_2_1 串的普通模式匹配 (暴力匹配 / BF算法)
 *
 * 思路: 从主串S的每个位置开始，逐个字符与模式串T比较，
 *       失配则回溯(i回到本次起始位置+1, j归零)。
 * 时间复杂度: O(n*m)，其中 n = lenS, m = lenT
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int StrLength(const char* S){
    int len = 0;
    while(S[len] != '\0') len++;
    return len;
}

int Index(const char* S, const char* T){
    int lenS = StrLength(S);
    int lenT = StrLength(T);

    if(lenT == 0) return 0;        // 空串是任何串的子串
    if(lenT > lenS) return -1;     // 模式串比主串长，不可能匹配

    int i = 0, j = 0;
    while(i < lenS && j < lenT){
        if(S[i] != T[j]){
            i = i - j + 1;  // 回溯: i回到本轮起始的下一个位置
            j = 0;          // j归零
        } else {
            i++;            // 匹配时双双前进
            j++;
        }
    }
    if(j == lenT){
        return i - j;       // 匹配成功，返回起始下标
    } else {
        return -1;
    }
}

int main(){
    printf("========== 暴力模式匹配测试 ==========\n\n");

    // 测试1: 正常匹配
    printf("Index(\"HelloWorld\", \"World\") = %d  (期望 5)\n",
           Index("HelloWorld", "World"));
    printf("Index(\"HelloWorld\", \"Hello\") = %d  (期望 0)\n",
           Index("HelloWorld", "Hello"));

    // 测试2: 不存在
    printf("Index(\"HelloWorld\", \"xyz\")   = %d  (期望 -1)\n",
           Index("HelloWorld", "xyz"));

    // 测试3: 空模式串
    printf("Index(\"HelloWorld\", \"\")       = %d  (期望 0)\n",
           Index("HelloWorld", ""));

    // 测试4: 模式串比主串长
    printf("Index(\"ab\", \"abcd\")           = %d  (期望 -1)\n",
           Index("ab", "abcd"));

    // 测试5: 相等
    printf("Index(\"abc\", \"abc\")           = %d  (期望 0)\n",
           Index("abc", "abc"));

    // 测试6: 重复字符——验证回溯逻辑正确性
    // "ababc"中找"abc"，正确应该在位置2
    printf("Index(\"ababc\", \"abc\")         = %d  (期望 2)\n",
           Index("ababc", "abc"));

    // 测试7: "mississippi"中找"issip"
    // m i s s i s s i p p i
    // 0 1 2 3 4 5 6 7 8 9 10
    // "issip"在位置4
    printf("Index(\"mississippi\", \"issip\") = %d  (期望 4)\n",
           Index("mississippi", "issip"));

    printf("\n========== 测试完成 ==========\n");
    return 0;
}
