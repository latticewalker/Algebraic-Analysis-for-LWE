p := 3329;
n := 6;
Fp := GF(p);
S<[s]> := PolynomialRing(Fp, n);
names := [ "s" cat "[" cat IntegerToString(i) cat "]" : i in [1..n] ];
AssignNames(~S, names);

// 为每个变元定义理想关系
I_relations := [];
for i in [1..n] do
    Append(~I_relations, s[i]^3 - s[i]);
end for;

I := ideal<S | I_relations>;
Q<[s]> := quo<S | I>;

file_path := "new_lwe_samples(a,b,e).m";  // 请替换为实际文件路径

// 初始化空列表
A_rows := [];
b_list := [];
e_list := [];

// 打开并读取文件
file := Open(file_path, "r");
while true do
    line := Gets(file);
    if IsEof(line) then break; end if;
    
    // 分割行数据
    data_str := Split(line, ",");
    
    // 转换为整数
    data := [StringToInteger(s) : s in data_str];
    
    // 提取 A_i (前n个元素)
    A_i := [data[i] : i in [1..n]];
    Append(~A_rows, A_i);
    
    // 提取 b_i (第n+1个元素)
    Append(~b_list, data[n+1]);
    
    // 提取 e_i (第n+2个元素，即最后一个)
    Append(~e_list, data[n+2]);
end while;

// 创建矩阵和向量
A := Matrix(A_rows);
b := Vector(b_list);
e := Vector(e_list);

// 计算乘积方程

function SelectMultipleRandomSubsets(A, b, e, subset_size, num_subsets)
    m := NumberOfRows(A);
    total := subset_size * num_subsets;
    
    // 检查是否有足够样本
    if total gt m then
        error "需要的总索引数 (", total, ") 超过可用样本数 (", m, ")";
    end if;

    // 不放回抽取 total 个不同索引
    chosen := [];
    available := [1..m];
    for i in [1..total] do
        pos := Random(1, #available);
        idx := available[pos];
        Append(~chosen, idx);
        // 从可用列表中移除已选索引
        available := [available[j] : j in [1..#available] | j ne pos];
    end for;

    // 分割成 num_subsets 组，每组 subset_size 个索引
    A_subs := [];
    b_subs := [];
    e_subs := [];
    all_indices := [];
    for k in [1..num_subsets] do
        start_idx := (k-1)*subset_size + 1;
        end_idx := k*subset_size;
        indices := chosen[start_idx .. end_idx];
        
        // 提取子矩阵和子向量
        A_sub := Matrix([Eltseq(A[i]) : i in indices]);
        b_sub := Vector([b[i] : i in indices]);
        e_sub := Vector([e[i] : i in indices]);
        
        Append(~A_subs, A_sub);
        Append(~b_subs, b_sub);
        Append(~e_subs, e_sub);
        Append(~all_indices, indices);
    end for;

    return A_subs, b_subs, e_subs, all_indices;
end function;

mul := 700;
num_subsets := 6;  // 生成 n 个子集

A_subs, b_subs, e_subs, all_indices := SelectMultipleRandomSubsets(A, b, e, mul, num_subsets);
// 校验所有索引是否不重复

// 验证每个子集使用不同的零分量
printf "生成了 %o 个子集，每个子集大小: %o\n", #A_subs, mul;
for i in [1..#A_subs] do
    zero_count := #[j : j in [1..mul] | e_subs[i][j] eq 0];
    printf "子集 %o: 使用零分量索引 %o, 子集中零分量数量: %o\n", 
           i, all_indices[i][1], zero_count;
end for;

Polys := [];
for t in [1..num_subsets] do
    poly_mul:=Q!1;
    for i in [1..mul] do
        linear_term := Q!0;
        for j in [1..n] do
            linear_term +:= (A_subs[t][i,j] mod p) * s[j];
        end for;
        poly := (b_subs[t][i] mod p) - linear_term;
        //time poly_mul := poly_mul * poly;
        poly_mul := poly_mul * poly;
    end for;
    Append(~Polys, poly_mul);
end for;




Poly_state:=[[Q!0 : i in [1..n]] : j in [1..n]];
for i in [1..num_subsets] do
    Poly_state[1][i]:=Polys[i];
end for;

for i in [1..#I_relations-1] do
    printf "正在消去变量 s[%o]...\n", i;
    for j in [1..#Polys] do
        Poly_state[i+1][j]:=Q!Resultant(S!Poly_state[i][j], S!I_relations[i], S!s[i]);
    end for;
end for;

// 定义递归回溯函数，增加 depth 参数用于缩进输出
function BackSubstitute(level, vals, Polys, Poly_state, n, Fp, S, Q, depth)
    // 根据深度生成缩进字符串
    indent := "";
    for i in [1..depth] do
        indent *:= "  ";
    end for;
    
    if level eq 0 then
        // 所有变元已赋值，验证是否满足所有原始乘积多项式
        for j in [1..num_subsets] do
            p := S!Polys[j];
            assign := [S!vals[i] : i in [1..n]];
            val := Evaluate(p, assign);
            if val ne 0 then
                return [];
            end if;
        end for;
        return [vals];
    end if;
    
    // 构建赋值列表：高于 level 的变元替换为具体值，level 及以下保留为变量
    assign := [S.i : i in [1..n]];
    for i in [level+1..n] do
        assign[i] := S!vals[i];
    end for;
    
    // 收集所有多项式在当前赋值下的根集（转换为整数表示）
    root_sets := [];
    for j in [1..n] do
        p := S!Poly_state[level][j];
        q := Evaluate(p, assign);   // q 现在只依赖于 s_level
        up := UnivariatePolynomial(q);
        if IsZero(up) then
            // 多项式恒为零，所有可能的 s_level 值（满足关系）都允许
            roots := [0, 1, 3328];   // 3328 = -1 mod 3329
        else
            // 取根并转换为整数，同时过滤满足 s_i^3 = s_i 的根
            roots := [IntegerRing()!r[1] : r in Roots(up) | r[1]^3 eq r[1]];
        end if;
        Append(~root_sets, roots);
    end for;
    
    if #root_sets eq 0 then 
        
        return []; 
    end if;
    
    // 取所有根集的交集
    common := root_sets[1];
    for i in [2..#root_sets] do
        common := [r : r in common | r in root_sets[i]];
        if #common eq 0 then 
            
            return []; 
        end if;
    end for;
    
    
    
    // 对每个公共根递归求解下一层
    solutions := [];
    for r in common do
        // 显式复制 vals，避免引用共享
        new_vals := [vals[i] : i in [1..n]];
        new_vals[level] := r;
        sols := BackSubstitute(level-1, new_vals, Polys, Poly_state, n, Fp, S, Q, depth+1);
        solutions cat:= sols;
    end for;
    return solutions;
end function;

// 初始调用（level = n，所有变元未定，vals 占位）
initial_vals := [0 : i in [1..n]];
all_solutions := BackSubstitute(n, initial_vals, Polys, Poly_state, n, Fp, S, Q, 0);

// 输出所有解
print "找到的所有解:";
for sol in all_solutions do
    print sol;
end for;