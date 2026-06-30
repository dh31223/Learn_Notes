# Flutter Learn Notes

## 1. 基础数据类型

### 1.1 var 可变变量

```dart
void main(){
  // var表示可变变量，所以age可以被修改，但是第一次复制之后，数据的类型无法再进行修改
  var age = 20;
  print(age);
  age = 21;
  print(age);
}
```

### 1.2 const 不可变变量（编译时确定值）

```dart
void main(){
  const pi = 3.14;
  print(pi);
  // 此时将会报错，因为pi无法被更改
  pi = 2;
  print(pi)
}

```

**注意：**

**不能把var赋值给const**
```dart
void main(){
  var a = 3.14159
  // 此时a是var，如果把a赋值给b的话，b是不可变的而a是可变的，会冲突
  const b = a
}
```


### 1.3 final 不可变变量（运行时确定值）


**final和const一样都是不可变的变量，但是final是在运行时确定值**

### 1.4 final和const的对比

**const在编译时，必须有具体的值，但是final在编译时可以没有具体的值，而是运行后再赋值。**

```dart
void main(){
  // DateTime.now()并不是一个值，而是一个会返回时间的函数，所以此时time并没有具体的值，但是final向编译器保证，以后会有一个不可更改的值给我。
  final time = DateTime.now();
  print(time);
  // 同理，此时c_time也没有具体的值，所以编译器不会让c_time通过编译
  const c_time = DateTime.now();
  print(c_time);
}
```

| 特性 | `const` | `final` |
|------|---------|---------|
| 赋值时机 | 编译时必须有值 | 运行时首次赋值 |
| 能否调用函数赋值 | ❌ 不能 `const x = someFunc()` | ✅ 可以 `final x = someFunc()` |
| 能否修改 | ❌ 不可变 | ❌ 不可变（与 const 相同） |
| 编译器行为 | 值写死进产物，类似 C 的 `#define` | 运行时分配内存，只赋值一次 |
| 适用场景 | 已知常量：`pi = 3.14`、颜色、字符串字面量 | 运行时才能确定的值：时间、用户输入、函数返回值 |
| 能否修饰对象 | ✅ 对象及内部全部不可变 | ✅ 对象本身不可变，但内部可以变 |
| 编译时计算 | ✅ `1 + 2`、`"a" + "b"` 都行 | N/A（运行时才赋值，不参与编译） |
| 内存 | 同一个 `const` 值全局只有一份 | 每个 `final` 变量独立分配内存 |


### 1.5 String 类型

1. String 是可变字符串变量
```dart
void main(){
  String str = "Hello World";
  print(str);
  // String 是可变字符串变量
  str = "Hello Flutter";
  print(str);
}
```
2. String 支持模板字符串

```dart
void main(){
  var time = 10;
  // 输出我要在10点吃早餐，类似于pythonf"{}"
  String text = "我要在${time}点吃早餐";
  print(text);
}
```

### 1.6 数字类型



