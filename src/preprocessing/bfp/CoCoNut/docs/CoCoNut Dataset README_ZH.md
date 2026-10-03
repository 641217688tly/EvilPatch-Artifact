### 训练数据发布

本次发布包含用于训练不同 CoCoNuT 模型的数据。
这些数据集已使用 lrztar 进行压缩。要解压数据，请使用 `lrzuntar`。

每种语言（Python、C、Java 和 JavaScript）都有一个压缩包。
除了 Java 有两个具有不同最大日期的训练数据集外，每种语言都有一个训练集。

这些数据集包含从 GitHub、GitLab 和 Bitbucket 提取的原始数据，尚未进行打乱或分词处理。

每个压缩包的组织结构如下：
`language/cutting_year/{add,rem,context,meta}.txt`

cutting_year（截止年份）是数据集中最新提交的年份。
它是根据测试基准（Defects4J、QuixBugs、BugAid、CodeFlaws 和 ManyBugs）中最早的 bug 日期决定的。
我们这样做是为了确保训练中不使用未来的数据（详见论文）。

每个数据集包含 4 个文件：
`add.txt`、`rem.txt`、`context.txt` 和 `meta.txt`

对于每个训练集，这 4 个文件之间存在逐行映射关系。

例如：

rem.txt 的前 5 行（即有 bug 的代码行/代码块）：

```
1 public synchronized StringBuffer append(char ch)
2 ensureCapacity_unsynchronized(count + 1); value[count++] = ch; return this;
3 public String substring(int beginIndex, int endIndex)
4 if (beginIndex < 0 || endIndex > count || beginIndex > endIndex) throw new StringIndexOutOfBoundsException(); if (beginIndex == 0 && endIndex == count) return this; int len = endIndex - beginIndex;  return new String(value, beginIndex + offset, len, (len << 2) >= value.length);
5 public Object next() {
```

add.txt 的前 5 行（即修复后的代码行/代码块）：

```
1 public StringBuffer append(Object obj)
2 return append(obj == null ? "null" : obj.toString());
3 public String substring(int begin)
4 return substring(begin, count);
5 public FSEntry next() {
```

这对应于以下 5 个实例：

```
- public synchronized StringBuffer append(char ch)
+ public StringBuffer append(Object obj)
```

```
- ensureCapacity_unsynchronized(count + 1); value[count++] = ch; return this;
+ return append(obj == null ? "null" : obj.toString());
```

```
- public String substring(int beginIndex, int endIndex)
+ public String substring(int begin)
```

```
- if (beginIndex < 0 || endIndex > count || beginIndex > endIndex) throw new StringIndexOutOfBoundsException(); if (beginIndex == 0 && endIndex == count) return this; int len = endIndex - beginIndex;  return new String(value, beginIndex + offset, len, (len << 2) >= value.length);
+ return substring(begin, count);
```

```
-public Object next() {
+public FSEntry next() { 
```

`context.txt` 包含相关的"上下文"。我们将上下文定义为（内联的）有 bug 的函数（包括有 bug 的代码行和注释）。
例如，

```
public synchronized StringBuffer append(char ch)
```

的上下文是其相关函数：

```
public synchronized StringBuffer append(char ch)  {    ensureCapacity_unsynchronized(count + 1);    value[count++] = ch;    return this;  }
```

此上下文用作上下文感知网络的辅助输入。FConv 模型不使用此上下文。

`meta.txt` 包含有关项目的一些元数据：

```
1056	/local/tlutelli/issta_data/temp/all_java0context/java/2006_temp/2006/1056/68a6301301378680519f2b146daec37812a1bc22/StringBuffer.java/buggy/core/src/classpath/java/java/lang/StringBuffer.java
```

`1056` 是项目 ID。`/local/...` 是有 bug 文件的绝对路径。可以解析该路径以提取提交 ID：`68a6301301378680519f2b146daec37812a1bc22`、文件名：`StringBuffer.java` 以及项目内的原始路径
`core/src/classpath/java/java/lang/StringBuffer.java`

遗憾的是，我们没有保留项目 ID 与 GitHub 仓库之间的映射关系。

### 关于 C 数据集的说明：

C语言数据集共包含2735506条数据


### 关于 Java 2010 和集成模型的说明：

对于 2010 Java 数据集，有超过 1400 万个实例。因此，在为该训练集构建集成方法时，我们将其随机分割为 10 个子数据集，并训练了 10 个具有不同训练输入的模型。
对于其他训练集，在训练所有集成模型时使用了整个训练集。