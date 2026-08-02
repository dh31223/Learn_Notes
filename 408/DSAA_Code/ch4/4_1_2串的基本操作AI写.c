/*
 * 4_1_2 串的基本操作 (408风格)
 *
 * 说明：408考试中，串直接用 char* (以'\0'结尾的字符数组) 表示，
 *       不需要定义复杂的存储结构体。
 *       需要手动实现的操作就这几个，核心是 Index() 模式匹配。
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

// ==================== 1. 求串长 ====================
int StrLength(const char *S) {
    int len = 0;
    while (S[len] != '\0') len++;
    return len;
}

// ==================== 2. 判空 ====================
bool StrEmpty(const char *S) {
    return S[0] == '\0';
}

// ==================== 3. 比较 ====================
// 返回值: S > T 返回正数; S = T 返回0; S < T 返回负数
int StrCompare(const char *S, const char *T) {
    int i = 0;
    while (S[i] != '\0' && T[i] != '\0') {
        if (S[i] != T[i])
            return S[i] - T[i];
        i++;
    }
    return S[i] - T[i];  // 谁先走到'\0'谁更短
}

// ==================== 4. 赋值 ====================
// 将 chars 的内容复制到新串 T 中返回
char *StrAssign(const char *chars) {
    int len = StrLength(chars);
    char *T = (char *)malloc((len + 1) * sizeof(char));
    for (int i = 0; i <= len; i++)  // <= 确保 '\0' 也复制
        T[i] = chars[i];
    return T;
}

// ==================== 5. 复制 ====================
char *StrCopy(const char *S) {
    return StrAssign(S);  // 和赋值逻辑一样
}

// ==================== 6. 求子串 ====================
// 返回 S 中从 pos 开始、长度为 len 的子串
// pos 从 0 开始计数
char *SubString(const char *S, int pos, int len) {
    int sLen = StrLength(S);
    if (pos < 0 || pos >= sLen || len < 0) {
        printf("SubString: 参数非法!\n");
        return NULL;
    }
    // 实际能取的长度不超过剩余长度
    if (pos + len > sLen)
        len = sLen - pos;

    char *Sub = (char *)malloc((len + 1) * sizeof(char));
    for (int i = 0; i < len; i++)
        Sub[i] = S[pos + i];
    Sub[len] = '\0';
    return Sub;
}

// ==================== 7. 拼接 ====================
// 将 S1 和 S2 拼接成新串 T 返回
char *Concat(const char *S1, const char *S2) {
    int len1 = StrLength(S1);
    int len2 = StrLength(S2);
    char *T = (char *)malloc((len1 + len2 + 1) * sizeof(char));

    for (int i = 0; i < len1; i++) T[i] = S1[i];
    for (int i = 0; i < len2; i++) T[len1 + i] = S2[i];
    T[len1 + len2] = '\0';
    return T;
}

// ==================== 8. 定位(模式匹配) ====================
// 返回子串 T 在主串 S 中首次出现的位置(下标从0开始)
// 未找到返回 -1
// 这是408算法题的核心考点!!!
int Index(const char *S, const char *T) {
    int n = StrLength(S);
    int m = StrLength(T);

    if (m == 0) return 0;  // 空串是任何串的子串

    // 暴力匹配: 从 S 的每个位置开始尝试匹配 T
    for (int i = 0; i <= n - m; i++) {
        int j;
        for (j = 0; j < m; j++) {
            if (S[i + j] != T[j])
                break;
        }
        if (j == m)  // 全部匹配成功
            return i;
    }
    return -1;
}

// ==================== 9. 清空 ====================
void ClearString(char *S) {
    S[0] = '\0';
}

// ==================== 10. 销毁 ====================
// 释放动态分配的串空间
void DestroyString(char **S) {
    free(*S);
    *S = NULL;
}

// ==================== 测试 ====================
int main() {
    // --- 测试 StrAssign / StrLength / StrEmpty ---
    char *s1 = StrAssign("Hello");
    char *s2 = StrAssign("HelloWorld");
    printf("s1 = \"%s\", 长度 = %d\n", s1, StrLength(s1));
    printf("s2 = \"%s\", 长度 = %d\n", s2, StrLength(s2));

    // --- 测试 StrCompare ---
    printf("StrCompare(s1, s2) = %d  (s1 < s2, 应为负数)\n", StrCompare(s1, s2));
    printf("StrCompare(s2, s1) = %d  (s2 > s1, 应为正数)\n", StrCompare(s2, s1));

    // --- 测试 SubString ---
    char *sub = SubString(s2, 5, 5);  // 从下标5开始取5个字符 -> "World"
    printf("SubString(s2, 5, 5) = \"%s\"\n", sub);

    // --- 测试 Concat ---
    char *s3 = Concat(s1, sub);  // "Hello" + "World" = "HelloWorld"
    printf("Concat(\"%s\", \"%s\") = \"%s\"\n", s1, sub, s3);
    printf("StrCompare(s2, s3) = %d  (s2 == s3, 应为0)\n", StrCompare(s2, s3));

    // --- 测试 Index (模式匹配) ---
    printf("Index(\"%s\", \"%s\") = %d  (\"World\"在\"HelloWorld\"中从下标5开始)\n",
           s2, sub, Index(s2, sub));
    printf("Index(\"%s\", \"%s\") = %d  (\"abc\"不在\"HelloWorld\"中, 应为-1)\n",
           s2, "abc", Index(s2, "abc"));

    // --- 测试 ClearString ---
    ClearString(s3);
    printf("清空后 s3 = \"%s\", StrEmpty(s3) = %s\n",
           s3, StrEmpty(s3) ? "true" : "false");

    // --- 清理 ---
    DestroyString(&s1);
    DestroyString(&s2);
    DestroyString(&sub);
    DestroyString(&s3);

    return 0;
}
