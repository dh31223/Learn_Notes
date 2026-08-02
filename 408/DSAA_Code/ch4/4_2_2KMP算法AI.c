/*
 * 4_2_2 KMP算法 (含 get_next 详解)
 *
 * 核心问题: 暴力匹配失配时, i 回退太多。KMP 让 i 只进不退,
 *          模式串 T 通过 next 数组决定 j 跳到哪里。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ==================== StrLength ====================
int StrLength(const char* S) {
    int len = 0;
    while (S[len] != '\0') len++;
    return len;
}

// ==================== get_next ====================
// next[j] 的含义:
//   模式串 T 在位置 j 失配时, j 应该跳回的位置。
//   即 T[0..j-1] 的最长相等前后缀的长度。
//
// 递推过程:
//   已知 next[0..j], 求 next[j+1]。
//   k = next[j] 表示 "T[0..k-1] == T[j-k..j-1]" (前面 k 个 == 后面 k 个)
//
//   情况1: T[j] == T[k]
//     → 前后缀可以再扩展一个字符, next[j+1] = k + 1
//
//   情况2: T[j] != T[k]
//     → 不能直接扩展, 需要找一个更短的相等前后缀。
//       由于已知 T[0..k-1] == T[j-k..j-1],
//       T[0..k-1] 的最长相等前后缀长度 = next[k],
//       它同时也是 T[j-k..j-1] 的(尾部)相等前后缀。
//       所以令 k = next[k], 回到情况1继续比较。
//       (这就叫 "用 next 来求 next" —— 递归/自引用)

void get_next(const char* T, int next[]) {
    int lenT = StrLength(T);
    int j = 0, k = -1;
    next[0] = -1;  // 约定: next[0] = -1, 表示 T[0] 失配时 i 右移

    while (j < lenT - 1) {
        if (k == -1 || T[j] == T[k]) {
            j++;
            k++;
            next[j] = k;   // 情况1: 扩展前后缀
        } else {
            k = next[k];   // 情况2: 找更短的相等前后缀 (理解难点!)
        }
    }
}

// ==================== KMP 模式匹配 ====================
// 思路: i 从不回退, 失配时只调整 j = next[j]
//       当 j == -1 时, 表示 T[0] 都不匹配, i 和 j 同时右移

int Index_KMP(const char* S, const char* T) {
    int lenS = StrLength(S);
    int lenT = StrLength(T);

    if (lenT == 0) return 0;
    if (lenT > lenS) return -1;

    int next[lenT];       // C99 VLA, 408考试可以这样写
    get_next(T, next);

    int i = 0, j = 0;
    while (i < lenS && j < lenT) {
        if (j == -1 || S[i] == T[j]) {
            i++;
            j++;
        } else {
            j = next[j];  // i 不动! 只调整 j
        }
    }

    if (j == lenT)
        return i - j;
    else
        return -1;
}

// ==================== get_nextval (优化版) ====================
// 问题: 若 T[j] == T[next[j]], 回退后再比较必然再次失配, 白费一次。
// 修正: 若 T[j] == T[next[j]], 则 nextval[j] = nextval[next[j]] (一路跳过)

void get_nextval(const char* T, int nextval[]) {
    int lenT = StrLength(T);
    int j = 0, k = -1;
    nextval[0] = -1;

    while (j < lenT - 1) {
        if (k == -1 || T[j] == T[k]) {
            j++;
            k++;
            // 关键一步: 如果字符相同, 直接继承前面的 nextval
            if (T[j] != T[k])
                nextval[j] = k;
            else
                nextval[j] = nextval[k];
        } else {
            k = nextval[k];
        }
    }
}

// ==================== 测试 / 演示 ====================
int main() {
    printf("========== KMP 算法测试 ==========\n\n");

    // 测试用例
    const char* test_cases[][2] = {
        {"HelloWorld",  "World"},
        {"HelloWorld",  "Hello"},
        {"ababcabcacbab", "abcac"},   // 经典KMP例子
        {"mississippi", "issip"},
        {"aaaaab",      "aaab"},
        {"ababa",       "aba"},       // 有重叠的模式
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int t = 0; t < num_tests; t++) {
        const char* S = test_cases[t][0];
        const char* T = test_cases[t][1];
        int lenT = StrLength(T);

        int next[lenT], nextval[lenT];
        get_next(T, next);
        get_nextval(T, nextval);

        int pos_bf = -1, pos_kmp = -1;
        // BF: 直接手写暴力匹配做对照
        {
            int ns = StrLength(S), nt = lenT;
            for (int i = 0; i <= ns - nt; i++) {
                int j;
                for (j = 0; j < nt; j++)
                    if (S[i + j] != T[j]) break;
                if (j == nt) { pos_bf = i; break; }
            }
        }
        pos_kmp = Index_KMP(S, T);

        // 打印 next 数组
        printf("T = \"%s\"\n", T);
        printf("  next:    ");
        for (int i = 0; i < lenT; i++) printf("%3d", next[i]);
        printf("\n  nextval: ");
        for (int i = 0; i < lenT; i++) printf("%3d", nextval[i]);
        printf("\n  BF=%d, KMP=%d", pos_bf, pos_kmp);
        printf("  %s\n\n", (pos_bf == pos_kmp) ? "✓" : "✗ 不一致!");
    }

    // ============ 手工推演: T = "abcac" ============
    printf("========== 手工推演 get_next(\"abcac\") ==========\n\n");
    {
        const char* T = "abcac";
        int lenT = StrLength(T);
        int next[lenT];

        // 逐步推演
        next[0] = -1;
        int j = 0, k = -1;

        printf("初始: next[0] = -1, j = 0, k = -1\n\n");

        int step = 0;
        while (j < lenT - 1) {
            printf("--- 第%d步 ---\n", ++step);
            printf("  比较 T[%d]='%c' 和 ", j, T[j]);
            if (k == -1)
                printf("k=-1(哨兵)\n");
            else
                printf("T[%d]='%c'\n", k, T[k]);

            if (k == -1 || T[j] == T[k]) {
                j++; k++;
                next[j] = k;
                printf("  → 匹配/哨兵, j=%d, k=%d, next[%d]=%d\n\n", j, k, j, k);
            } else {
                printf("  → 不匹配, k = next[%d] = %d\n\n", k, next[k]);
                k = next[k];
            }
        }

        printf("最终 next 数组: ");
        for (int i = 0; i < lenT; i++) printf("%d ", next[i]);
        printf("\n");
    }

    return 0;
}
