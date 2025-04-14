QA3 = """
<instruction>what is the vim command to supersede 'C_buffer[0][0]' with 'C00'?</instruction>
<response>To supersede `'C_buffer'` with `'C00'` in Vim, you can use the substitute command. Here's how you can do it:

1. **Press `Esc`** to ensure you are in normal mode.
2. **Type the following command** and press `Enter`:

   ```vim
   :%s/C_buffer\[\d\+\]\[\d\+\]/C00/g
   ```

   This command will replace all occurrences of `'C_buffer[any_number][any_number]'` with `'C00'` across the entire file.

If you want to confirm each replacement, you can add the `c` flag at the end:

```vim
:%s/C_buffer\[\d\+\]\[\d\+\]/C00/gc
```
</response>
"""

QA1 = """
<instruction>what is the vim command to supersede 'C_buffer[0][0]' with 'C00'?</instruction>
<response>To supersede `'C_buffer'` with `'C00'` in Vim, you can use the substitute command. Here's how you can do it:

1. **Press `Esc`** to ensure you are in normal mode.
2. **Type the following command** and press `Enter`:

   ```vim
   :%s/C_buffer\[\d\+\]\[\d\+\]/C00/g
   ```

   This command will replace all occurrences of `'C_buffer[any_number][any_number]'` with `'C00'` across the entire file.

If you want to confirm each replacement, you can add the `c` flag at the end:

```vim
:%s/C_buffer\[\d\+\]\[\d\+\]/C00/gc
```

This will prompt you to confirm each replacement with a `y` for yes or `n` for no. 

# Explanation:
- **`:%s/`**: This part of the command tells Vim to search and replace across the entire file (`%`).
- **`C_buffer$$\d\+$$$$\d\+$$`**: This is the pattern to match. It looks for `'C_buffer'` followed by any number of digits (`\d\+`) enclosed in square brackets (`$$\d\+$$`) twice.
- **`/C00/`**: This is the replacement string.
- **`g`**: This flag tells Vim to replace all occurrences in each line.
- **`c` (optional)**: This flag prompts for confirmation before each replacement.

Citations:
[1] https://linuxize.com/post/vim-find-replace/
[2] https://superuser.com/questions/464203/vim-c-shortcut
[3] https://stackoverflow.com/questions/19994922/find-and-replace-strings-in-vim-on-multiple-lines
[4] https://dev.to/iggredible/a-faster-vim-workflow-with-buffers-and-args-51kf
[5] https://www.ubuntumint.com/search-replace-text-vim-editor/
[6] https://linuxhandbook.com/vim-buffers/
[7] https://vim.blue/vim-replace-string-with-confirmation/
[8] https://vim.rtorr.com

---
Answer from Perplexity: pplx.ai/share 
</response>
"""


QA2 = """<instruction>What is the probability of getting a sum of 7 when rolling two dice?</instruction> <response>The probability of getting a sum of 7 when rolling two dice is 1/6.</response> """


QA4 = """
<instruction>What is tool-calling of LLMs?
<response>
Tool use is a curious feature – everyone thinks it’s great, but most people haven’t tried it themselves. Conceptually, it’s very straightforward: you give some tools (callable functions) to your LLM, and it can decide to call them to help it respond to user queries. Maybe you give it a calculator so it doesn’t have to rely on its internal, unreliable arithmetic abilities. Maybe you let it search the web or view your calendar, or you give it (read-only!) access to a company database so it can pull up information or search technical documentation.

Tool use overcomes a lot of the core limitations of LLMs. Many LLMs are fluent and loquacious but often imprecise with calculations and facts and hazy on specific details of more niche topics. They don’t know anything that happened after their training cutoff date. They are generalists; they arrive into the conversation with no idea of you or your workplace beyond what you give them in the system message. Tools provide them with access to structured, specific, relevant, and up-to-date information that can help a lot in making them into genuinely helpful partners rather than just fascinating novelty.


</response>
"""


QA5 = """
<instruction>
Is there any avx intrinsics to fuse the multiply and add operation of integer, like fma?
</instruction>
<response>
Unfortunately, **AVX2** (the version of AVX that supports integer operations) does not have a fused multiply-add (FMA) instruction for integers like it does for floating-point numbers (`_mm256_fmadd_ps` for floats). The FMA instruction set (`FMA3`) is primarily designed for floating-point operations and does not include integer equivalents.

For integer operations, you must perform the multiply and add operations separately:
1. Use `_mm256_mullo_epi32` for 32-bit integer multiplication (low 32 bits of the result).
2. Use `_mm256_add_epi32` for 32-bit integer addition.

#### Example of Separate Multiply and Add
```cpp
__m256i a = _mm256_set1_epi32(2); // Broadcast 2 across the vector
__m256i b = _mm256_loadu_si256((__m256i*)blockB); // Load B
__m256i c = _mm256_loadu_si256((__m256i*)C); // Load C

// Multiply and add
__m256i prod = _mm256_mullo_epi32(a, b); // Multiply
__m256i result = _mm256_add_epi32(prod, c); // Add
```

#### Why No Integer FMA?
The lack of integer FMA in AVX2 is likely due to:
1. **Use Case**: Floating-point FMA is critical for numerical accuracy in scientific computing, where rounding errors can accumulate. Integer arithmetic does not suffer from the same rounding issues, so FMA is less critical.
2. **Hardware Complexity**: Implementing FMA for integers would require additional hardware resources, which might not justify the performance gains for most integer workloads.

### Workarounds
If you need to optimize integer multiply-add operations:
1. **Manual Loop Unrolling**: Manually unroll loops to reduce overhead and improve instruction-level parallelism.
2. **Use 64-bit Accumulators**: If your integers are small enough, you can use 64-bit integers (`__m256i` with `_mm256_add_epi64` and `_mm256_mul_epi32`) to avoid overflow and improve performance in some cases.
3. **AVX-512**: If you have access to AVX-512, it provides more advanced integer operations, though still no integer FMA.

### Example with 64-bit Accumulators
If your integers are small enough to fit into 32 bits but you want to avoid overflow, you can use 64-bit accumulators:
```cpp
__m256i a = _mm256_set1_epi32(2); // Broadcast 2 across the vector
__m256i b = _mm256_loadu_si256((__m256i*)blockB); // Load B
__m256i c = _mm256_loadu_si256((__m256i*)C); // Load C

// Multiply and add with 64-bit intermediates
__m256i prod = _mm256_mul_epi32(a, b); // Multiply (produces 64-bit results)
__m256i result = _mm256_add_epi64(prod, _mm256_cvtepi32_epi64(c)); // Add
```

### Conclusion
While there is no direct integer FMA in AVX2, you can achieve similar results by combining `_mm256_mullo_epi32` and `_mm256_add_epi32`. If you need higher performance or are working with AVX-512, explore those extensions for additional integer operation support.
</response>
"""

# jumping heading level problem
QA6 = """
<instruction>
Is there any avx intrinsics to fuse the multiply and add operation of integer, like fma?
</instruction>
<response>
Unfortunately, **AVX2** (the version of AVX that supports integer operations) does not have a fused multiply-add (FMA) instruction for integers like it does for floating-point numbers (`_mm256_fmadd_ps` for floats). The FMA instruction set (`FMA3`) is primarily designed for floating-point operations and does not include integer equivalents.
For integer operations, you must perform the multiply and add operations separately:
1. Use `_mm256_mullo_epi32` for 32-bit integer multiplication (low 32 bits of the result).
2. Use `_mm256_add_epi32` for 32-bit integer addition.

### Example of Separate Multiply and Add
```cpp
__m256i a = _mm256_set1_epi32(2); // Broadcast 2 across the vector
__m256i b = _mm256_loadu_si256((__m256i*)blockB); // Load B
__m256i c = _mm256_loadu_si256((__m256i*)C); // Load C
</response>
"""


QA7 = """
<instruction>Give me an introduction of the game monster hunter.</instruction>
<response>
"Monster Hunter" is a popular action role-playing game (RPG) series developed and published by Capcom. The franchise debuted in 2004 and has since grown into a global phenomenon, known for its challenging gameplay, intricate mechanics, and immersive world-building.
Core Gameplay:

In "Monster Hunter," players take on the role of Hunters who are tasked with tracking, battling, and defeating or capturing large and often fearsome creatures known as monsters. These monsters inhabit diverse ecosystems, ranging from lush forests and arid deserts to frozen tundras and volcanic regions. The game emphasizes preparation, strategy, and skill, as players must study monster behavior, exploit weaknesses, and use the environment to their advantage.


</response>
"""


# few-shot example
fewshot_example = {
    "instruction": "What is tool-calling of Large Language Models?",
    "input": "",
    "output": """\
Tool use is a curious feature – everyone thinks it’s great, but most people haven’t tried it themselves. Conceptually, it’s very straightforward: you give some tools (callable functions) to your LLM, and it can decide to call them to help it respond to user queries. Maybe you give it a calculator so it doesn’t have to rely on its internal, unreliable arithmetic abilities. Maybe you let it search the web or view your calendar, or you give it (read-only!) access to a company database so it can pull up information or search technical documentation.
Tool use overcomes a lot of the core limitations of LLMs. Many LLMs are fluent and loquacious but often imprecise with calculations and facts and hazy on specific details of more niche topics. They don’t know anything that happened after their training cutoff date. They are generalists; they arrive into the conversation with no idea of you or your workplace beyond what you give them in the system message. Tools provide them with access to structured, specific, relevant, and up-to-date information that can help a lot in making them into genuinely helpful partners rather than just fascinating novelty.
"""
}

fewshot_response="""\
---
tags:
  - LLM LLM/tool-calling
---
# Instruction
What is tool-calling of Large Language Models?

# Summary
Tool-calling is an capability of LLM to call external tools to overcomes a lot of the core limitations of LLMs.

## Details
Tool use is a curious feature – everyone thinks it’s great, but most people haven’t tried it themselves.
Conceptually, it’s very straightforward: you give some tools (callable functions) to your LLM, and it can decide to call them to help it respond to user queries.
Maybe you give it a calculator so it doesn’t have to rely on its internal, unreliable arithmetic abilities. Maybe you let it search the web or view your calendar, or you give it (read-only!) access to a company database so it can pull up information or search technical documentation.
Tool use overcomes a lot of the core limitations of LLMs. Many LLMs are fluent and loquacious but often imprecise with calculations and facts and hazy on specific details of more niche topics. They don’t know anything that happened after their training cutoff date.
They are generalists; they arrive into the conversation with no idea of you or your workplace beyond what you give them in the system message. Tools provide them with access to structured, specific, relevant, and up-to-date information that can help a lot in making them into genuinely helpful partners rather than just fascinating novelty.
"""


QA7 = """\
<instruction>
To minimize control divergence, we need to design kernels to ensure that threads within the same warp follow the same execution path. This can be achieved by restructuring code to reduce branching, using data structures that ensure all threads follow similar execution paths, or employing techniques such as predication.  Show me an example of predication.
</instruction>
<response>
Predication is a technique used to minimize control divergence by converting conditional branches into arithmetic or logical operations that all threads execute. Instead of using an `if` statement that causes divergence, predication ensures that all threads execute the same instructions but selectively update results based on a condition.

Here’s a CUDA example illustrating predication:

### Without Predication (Causes Divergence)
```cpp
__global__ void without_predication(int *data, int threshold) {
    int idx = threadIdx.x;
    if (data[idx] > threshold) {  // Branching may cause divergence
        data[idx] *= 2;
    }
}
```
In the above code, if some threads within a warp evaluate `data[idx] > threshold` as true while others evaluate it as false, control divergence occurs.

---

### With Predication (Minimizes Divergence)
```cpp
__global__ void with_predication(int *data, int threshold) {
    int idx = threadIdx.x;
    int condition = data[idx] > threshold;  // Compute condition
    data[idx] *= (condition ? 2 : 1);  // No explicit branching
}
```
Here, every thread executes the same multiplication operation, but for threads where `condition` is false, the multiplication by `1` ensures the value remains unchanged. This removes control divergence and improves warp efficiency.

Would you like a more complex example, such as predicated memory writes?
</response>\
"""

QA8 = """\
<instruction>
excel, get unique value
</instruction>
<response>
To get unique values in Excel, you can use either the **UNIQUE function** or **Advanced Filter**. Here's how to do both:

#### Using the UNIQUE Function

The **UNIQUE function** is available in Excel 365 and Excel 2021. It returns a list of unique values from a specified range.

1. **Select the Cell**: Choose where you want to display the unique values.
2. **Enter the Formula**: Type `=UNIQUE(range)`, where "range" is the area from which you want to extract unique values (e.g., `=UNIQUE(A2:A10)`).
3. **Press Enter**: The formula will automatically spill the unique values into the cells below or beside it, depending on the orientation of your data.

**Example**:
```excel
=UNIQUE(A2:A10)
```

#### Using Advanced Filter

If you don't have access to the UNIQUE function, you can use **Advanced Filter**:

1. **Select the Data Range**: Highlight the column from which you want to extract unique values.
2. **Go to Data Tab**: Click on "Data" in the ribbon.
3. **Advanced Filter**: Click on "Sort & Filter" and select "Advanced."
4. **Check Unique Records Only**: Mark the checkbox for "Unique records only."
5. **Copy to Another Location**: Optionally, you can copy the unique values to another location by specifying a range in the "Copy to" field.
6. **Click OK**: Excel will display the unique values in the specified location.

#### Additional Tips

- **Sorting Unique Values**: To sort the unique values, you can nest the UNIQUE function within the SORT function:
  ```excel
  =SORT(UNIQUE(A2:A10))
  ```
- **Counting Unique Values**: Use the COUNTA function with UNIQUE to count the number of unique values:
  ```excel
  =COUNTA(UNIQUE(A2:A10))
  ```

Citations:
[1] https://www.ablebits.com/office-addins-blog/get-unique-distinct-values-excel/
[2] https://help.chi.ac.uk/excel-unique-function
[3] https://www.ablebits.com/office-addins-blog/excel-unique-function-find-unique-values/
[4] https://www.datacamp.com/tutorial/count-unique-values-excel
[5] https://www.indeed.com/career-advice/career-development/unique-values-excel
[6] https://support.microsoft.com/en-us/office/unique-function-c5ab87fd-30a3-4ce9-9d1a-40204fb85e1e
[7] https://support.microsoft.com/en-us/office/filter-for-unique-values-or-remove-duplicate-values-ccf664b0-81d6-449b-bbe1-8daaec1e83c2
[8] https://superuser.com/questions/49614/how-do-i-get-the-distinct-unique-values-in-a-column-in-excel

---
Answer from Perplexity: pplx.ai/share
</response>\
"""