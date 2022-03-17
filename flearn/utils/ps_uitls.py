from tqdm import trange, tqdm

def output_ps(ps):
    st = f"ps = ["
    for p in ps:
        st = f"{st}{p},"
    st = f"{st}]"
    tqdm.write(st)

def gini_gen(num):
    tsum = 0
    ps = []
    odd = 1
    for i in range(0, num):
        tsum  += odd
        ps.append(odd)
        odd += 2
    ps = [(p/tsum) for p in ps]
  
    return ps

def get_geometric_gen(z):
    def geometric_gen(num):
        nonlocal z
        tsum = 0
        ps = []
        val = 1.0
        for i in range(0, num):
            tsum  += val
            ps.append(val)
            val *= z
        ps = [(p/tsum) for p in ps]
        return ps
    return geometric_gen

def harmonic_gen(num):
    tsum = 0
    ps = []
    start = 1
    div = 1 
    val = 0
    for i in range(0, num):
        val = val + start/div
        tsum  += val
        ps.append(val)
        div += 1

    ps = [(p/tsum) for p in ps]
    return ps  