p := 3329;
n := 6;
Fp := GF(p);
S<[s]> := PolynomialRing(Fp, n);
names := [ "s" cat "[" cat IntegerToString(i) cat "]" : i in [1..n] ];
AssignNames(~S, names);

// Define ideal relations for each variable
I_relations := [];
for i in [1..n] do
    Append(~I_relations, s[i]^3 - s[i]);
end for;

I := ideal<S | I_relations>;
Q<[s]> := quo<S | I>;

file_path := "new_lwe_samples(a,b,e).m";  // please replace with actual file path

// Initialize empty lists
A_rows := [];
b_list := [];
e_list := [];

// Open and read file
file := Open(file_path, "r");
while true do
    line := Gets(file);
    if IsEof(line) then break; end if;
    
    // Split line data
    data_str := Split(line, ",");
    
    // Convert to integers
    data := [StringToInteger(s) : s in data_str];
    
    // Extract A_i (first n elements)
    A_i := [data[i] : i in [1..n]];
    Append(~A_rows, A_i);
    
    // Extract b_i (element n+1)
    Append(~b_list, data[n+1]);
    
    // Extract e_i (last element)
    Append(~e_list, data[n+2]);
end while;

// Create matrix and vectors
A := Matrix(A_rows);
b := Vector(b_list);
e := Vector(e_list);

// Compute product equations

function SelectMultipleRandomSubsets(A, b, e, subset_size, num_subsets)
    m := NumberOfRows(A);
    total := subset_size * num_subsets;
    
    // Check if enough samples
    if total gt m then
        error "Total required indices (", total, ") exceeds available samples (", m, ")";
    end if;

    // Draw total distinct indices without replacement
    chosen := [];
    available := [1..m];
    for i in [1..total] do
        pos := Random(1, #available);
        idx := available[pos];
        Append(~chosen, idx);
        // Remove selected index from available list
        available := [available[j] : j in [1..#available] | j ne pos];
    end for;

    // Split into num_subsets groups, each of size subset_size
    A_subs := [];
    b_subs := [];
    e_subs := [];
    all_indices := [];
    for k in [1..num_subsets] do
        start_idx := (k-1)*subset_size + 1;
        end_idx := k*subset_size;
        indices := chosen[start_idx .. end_idx];
        
        // Extract submatrix and subvectors
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

mul := 700; //K
num_subsets := 6;  // W

A_subs, b_subs, e_subs, all_indices := SelectMultipleRandomSubsets(A, b, e, mul, num_subsets);
// Verify all indices are distinct

// Verify that each subset uses a different zero component
printf "Generated %o subsets, each size: %o\n", #A_subs, mul;
for i in [1..#A_subs] do
    zero_count := #[j : j in [1..mul] | e_subs[i][j] eq 0];
    printf "Subset %o: using zero component indices %o, number of zero components in subset: %o\n", 
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
    printf "Eliminating variable s[%o]...\n", i;
    for j in [1..#Polys] do
        Poly_state[i+1][j]:=Q!Resultant(S!Poly_state[i][j], S!I_relations[i], S!s[i]);
    end for;
end for;

// Define recursive backtracking function, add depth parameter for indentation output
function BackSubstitute(level, vals, Polys, Poly_state, n, Fp, S, Q, depth)
    // Generate indentation string based on depth
    indent := "";
    for i in [1..depth] do
        indent *:= "  ";
    end for;
    
    if level eq 0 then
        // All variables assigned, verify all original product polynomials
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
    
    // Build assignment list: substitute concrete values for variables above level,
    // keep level and below as variables
    assign := [S.i : i in [1..n]];
    for i in [level+1..n] do
        assign[i] := S!vals[i];
    end for;
    
    // Collect root sets for all polynomials under current assignment (convert to integer representation)
    root_sets := [];
    for j in [1..n] do
        p := S!Poly_state[level][j];
        q := Evaluate(p, assign);   // q now depends only on s_level
        up := UnivariatePolynomial(q);
        if IsZero(up) then
            // Polynomial identically zero, all possible s_level values (satisfying relation) allowed
            roots := [0, 1, 3328];   // 3328 = -1 mod 3329
        else
            // Take roots, convert to integers, and filter those satisfying s_i^3 = s_i
            roots := [IntegerRing()!r[1] : r in Roots(up) | r[1]^3 eq r[1]];
        end if;
        Append(~root_sets, roots);
    end for;
    
    if #root_sets eq 0 then 
        
        return []; 
    end if;
    
    // Take intersection of all root sets
    common := root_sets[1];
    for i in [2..#root_sets] do
        common := [r : r in common | r in root_sets[i]];
        if #common eq 0 then 
            
            return []; 
        end if;
    end for;
    
    
    
    // For each common root, recursively solve next level
    solutions := [];
    for r in common do
        // Explicitly copy vals to avoid reference sharing
        new_vals := [vals[i] : i in [1..n]];
        new_vals[level] := r;
        sols := BackSubstitute(level-1, new_vals, Polys, Poly_state, n, Fp, S, Q, depth+1);
        solutions cat:= sols;
    end for;
    return solutions;
end function;

// Initial call (level = n, all variables undetermined, vals placeholder)
initial_vals := [0 : i in [1..n]];
all_solutions := BackSubstitute(n, initial_vals, Polys, Poly_state, n, Fp, S, Q, 0);

// Output all solutions
print "All solutions found:";
for sol in all_solutions do
    print sol;
end for;
