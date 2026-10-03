@CoCoNut Dataset READEME.md

`CoCoNut Dataset READEME.md`中记录了CoCoNut数据集的格式，以下是其`CoCoNut Dataset READEME.md`的内容：

````markdown
### Training Data Release

This release contains the data used to train the different CoCoNuT models.
The datasets have been compressed using lrztar. To extract the data, use
`lrzuntar`.

There is one archive per language (Python, C, Java, and JavaScript). 
There is one training set per language, except for Java where we have two different training 
dataset with two different max dates.

These datasets contains raw data extracted from GitHub, GitLab, and Bitbucket and have 
not been shuffled nore tokenized.

Each archive is ordered as:
`language/cutting_year/{add,rem,context,meta}.txt`

The cutting year is the year of the newest commit in the dataset. 
It was decided based on the date of the oldest bug in the test benchmark
(Defects4J, QuixBugs, BugAid, CodeFlaws, and ManyBugs).
We did this to ensure future data is not used in the training (see paper for more information).

Each dataset consists of 4 files:
`add.txt`, `rem.txt`, `context.txt`, and `meta.txt`

There is a line-to-line mapping between these 4 files for each training set.

For example:

5 first lines of rem.txt (i.e., the buggy line/hunk):
```
1 public synchronized StringBuffer append(char ch)
2 ensureCapacity_unsynchronized(count + 1); value[count++] = ch; return this;
3 public String substring(int beginIndex, int endIndex)
4 if (beginIndex < 0 || endIndex > count || beginIndex > endIndex) throw new StringIndexOutOfBoundsException(); if (beginIndex == 0 && endIndex == count) return this; int len = endIndex - beginIndex;  return new String(value, beginIndex + offset, len, (len << 2) >= value.length);
5 public Object next() {
```
5 first lines of add.txt (i.e., the fixed line/hunk):
```
1 public StringBuffer append(Object obj)
2 return append(obj == null ? "null" : obj.toString());
3 public String substring(int begin)
4 return substring(begin, count);
5 public FSEntry next() {
```

This maps to the 5 instances:
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

`context.txt` contains the associated "context". We call context the (in-lined) buggy function (including the buggy lines and comments).
For example, the context of 
```
public synchronized StringBuffer append(char ch)
```
is its associated function:
```
public synchronized StringBuffer append(char ch)  {    ensureCapacity_unsynchronized(count + 1);    value[count++] = ch;    return this;  }
```

This context is used as a secondary input to the context-aware network. The FConv model does not use this context.

`meta.txt` contains some metadata about the project:
```
1056	/local/tlutelli/issta_data/temp/all_java0context/java/2006_temp/2006/1056/68a6301301378680519f2b146daec37812a1bc22/StringBuffer.java/buggy/core/src/classpath/java/java/lang/StringBuffer.java
```
`1056` is the project id. `/local/...` is the absolute path to the buggy file. This can be parsed to extract the commit id: `68a6301301378680519f2b146daec37812a1bc22`, the file name: `StringBuffer.java` and the original path within the project 
`core/src/classpath/java/java/lang/StringBuffer.java`

Unfortunately, we do not kept the mapping between project id and GitHub repository.


### Note about Java 2010 and ensemble:
For the 2010 Java dataset, there are more than 14M instances. Therefore, when building an Ensemble 
approach for this training set, we splitted it into 10 random sub dataset and trained 10 models each with different training inputs.
For other training set. The entire training set was used during training of all ensemble models.
````



Task：我希望处理CoCoNut数据集并最终得到如下格式的jsonl数据，请你为我规划实现步骤，并逐步完成：

```jsonl
{
    "id": "/local/tlutelli/issta_data/temp/c/2005_temp/2005/10355/784e792a04c600b79d535ad81a5ba32b4f4de33e/vmeUniverse.c/clean/c/src/lib/libbsp/shared/vmeUniverse/vmeUniverse.c", // bug 文件的绝对路径
    "language": "C",
    "buggy_code": "漏洞代码，即context",
    "fixed_code": "在漏洞代码上应用了所有<remove, add>变更对后得到的修复后代码"
}
```



补充信息：

1. 根据`meta.txt`可知，当前数据集中存在大量针对同一文件的重复冗余的<`remove`, `add`>变更对，比如对于`/local/tlutelli/issta_data/temp/c/2005_temp/2005/10355/1be1e913564b73bf50ce1aa58c003e564ddae83a/erc32sonic.c/buggy/c/src/lib/libbsp/sparc/erc32/erc32sonic/erc32sonic.c`，其在第1行~2行的两个<`remove`, `add`>变更对与在第1367754~1367755行的两个<`remove`, `add`>变更对完全一致。经过验证，目前已经确定各文件中第1行到1367753行的内容与第1367754行到2735506行的内容重复，故对原始数据集采取了去重预处理。去重后的`add.txt`、`rem.txt`、`context.txt` 和 `meta.txt`文件被保存在了`<DATASET_ROOT>\CoCoNut\dataset\c\dedup`目录下（去重后的文件内均有1367753行，这四个文件共占用了10GB的存储空间，其中`context.txt`占据了9.63GB的存储空间）

3. 当前项目根目录的绝对路径为：`<DATASET_ROOT>\CoCoNut`

4. 当前项目所需的数据在：`<DATASET_ROOT>\CoCoNut\dataset\c`，数据按照以下结构组织：

   `<DATASET_ROOT>\CoCoNut\dataset\c`

   ├── `raw/`

   │  ├── add.txt

   │  ├── rem.txt

   │  ├── context.txt

   │  └── meta.txt

4. `context.txt`中任意一行的代码内都没有换行符`\n`



额外要求：

1. 在处理数据集前，请你将满足过滤条件1的数据进行删除以减少数据量（你可以将过滤后的数据存储在`<DATASET_ROOT>\CoCoNut\dataset\c\filtered`目录下）

2. 请将最终代码写入`<DATASET_ROOT>\CoCoNut\dataset\c\processed`目录下的`CoCoNut.jsonl`文件中

3. 请使用Jupyter Notebook来组织代码

4. 请将与业务逻辑无关的工具函数写入`utils.py`中，并在需要使用时导入Notebook

5. 相关代码需要有中断恢复功能：如果在数据处理过程中进程突然中断，在下次启动后，之前未完全写入到`CoCoNut.jsonl`中的残缺数据行需要被删除，同时脚本能够检查已经处理过的数据行并跳过它们继续向后处理

6. 考虑到数据量较为庞大（10GB），你可以自行决定是否使用mysql、sqlite等数据库存储数据以加速数据处理

7. 对于`rem.txt`中的`remove`代码段可能出现与`context.txt`中对应行的`context`代码段无法完全匹配的情况

   - 比如：
     1. `remove`: "unsigned32  regno, unsigned32  value"  (逗号后有2个空格)
     2. `context`: "unsigned32 regno, unsigned32 value"    (逗号后有1个空格)
   - 针对这一情况，你可能需要通过对`remove`和`context`采取应用正则表达式或标准化空格等方法来辅助匹配

8. 在CoCoNut数据集中，针对同一个文件（相同的`meta`路径）可能有多个不同函数的变更，每个函数对应不同的`context`

   - 比如对于`/local/tlutelli/issta_data/temp/c/2005_temp/2005/10355/1be1e913564b73bf50ce1aa58c003e564ddae83a/erc32sonic.c/buggy/c/src/lib/libbsp/sparc/erc32/erc32sonic/erc32sonic.c`：

     1. `meta.txt`中第1行-第2行的变更对针对的是`erc32_sonic_write_register`函数，其`context`为：

        ```c
        void erc32_sonic_write_register(  void       *base,  unsigned32  regno,  unsigned32  value){  volatile unsigned32 *p = base;#if (SONIC_DEBUG & SONIC_DEBUG_PRINT_REGISTERS)  printf( "%p Write 0x%04x to %s (0x%02x)\n",      &p[regno], value, SONIC_Reg_name[regno], regno );  fflush( stdout );#endif  p[regno] = 0x0ffff & value;}
        ```

     2. `meta.txt`中第155行-第157行的变更对针对的是`erc32_sonic_read_register`函数，其`context`为：

        ```c
        unsigned32 erc32_sonic_read_register(  void       *base,  unsigned32  regno){  volatile unsigned32 *p = base;  unsigned32           value;  value = p[regno];#if (SONIC_DEBUG & SONIC_DEBUG_PRINT_REGISTERS)  printf( "%p Read 0x%04x from %s (0x%02x)\n",      &p[regno], value, SONIC_Reg_name[regno], regno );  fflush( stdout );#endif  return 0x0ffff & value;
        ```

   - 针对这一情况，请你将(`meta_file_path`, `context中的函数名或context的哈希值`)作为主键，按照 (`file_path`, `context`) 进行分组隔离，为同一文件的每个 buggy 函数单独产出一条JSONL记录，而不是将它们合并进同一个`context`



以下是对噪音数据的过滤条件：

1. 如果`context`的字节长度超过n（n默认为75000），则舍弃该条数据
2. 对于应用了所有变更后的`context`，如果其字节长度超过n（n默认为75000），则舍弃该条数据
3. 对于针对同一个项目的同一个文件的一个或多个<`remove`, `add`>变更对，如果某个变更对的`remove`为空（即，移除`context`的某个空代码行，然后插入一段代码`add`），则舍弃该条数据（因为由于数据集信息的缺失，无法确定`remove`要移除具体哪个空行）
4. 对于针对同一个项目的同一个文件的一个或多个<`remove`, `add`>变更对，在对`context`应用变更的过程中如果有任意的`remove`在`context`中匹配到了多个代码段（比如<`remove`, `add`> = <“    }”, “”>，而`context`中的许多行都存在“    }”这段代码），则舍弃该条数据（因为由于数据集信息的缺失，无法确定`remove`要移除具体哪一个位置上的代码）
5. 对于针对同一个项目的同一个文件的一个或多个<`remove`, `add`>变更对，在对`context`应用变更的过程中如果有任意的`remove`没有在`context`中匹配到任何代码段（即，在使用了直接匹配、基于空格符删除后的匹配、基于正则表达式的匹配等匹配方法后依旧没有匹配到任何代码段），则舍弃该条数据（因为这可能会造成代码无法被完全修复）

