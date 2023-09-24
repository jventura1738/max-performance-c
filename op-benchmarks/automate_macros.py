NUM_REGISTERS = 32  # MUST BE DIV BY 2!
OP_TYPE = "SIMD_ADD_ASM"
PARENT_OP = OP_TYPE + "_CHAINS"
MACRO_SUFFIX = "\\ \n"

registers = [f"_{i}x" for i in range(NUM_REGISTERS)]
with open("benchmark.txt", "w") as fptr:
    for register in registers:
        fptr.write(f"__m256d {register} = " + "{1.0,1.0,1.0,1.0};\n")

    fptr.write("\n")

    fptr.write(f"#define {PARENT_OP}()" + MACRO_SUFFIX)
    for i in range(0, NUM_REGISTERS, 2):
        srcx, destx = registers[i], registers[i+1]
        fptr.write(f"\t{OP_TYPE}({srcx},{srcx},{destx});" + MACRO_SUFFIX)

    fptr.write("\n")

    i = 10
    while i < 10000:
        fptr.write(f"#define {PARENT_OP}_{i}()" + MACRO_SUFFIX)
        for _ in range(i, i*10, i):
            if i == 10:
                fptr.write(f"\t{PARENT_OP}();" + MACRO_SUFFIX)
            else:
                fptr.write(f"\t{PARENT_OP}_{i//10}();" + MACRO_SUFFIX)
        i *= 10

        fptr.write("\n")
    fptr.write("\n")

