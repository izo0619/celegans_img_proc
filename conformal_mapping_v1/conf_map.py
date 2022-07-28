import matlab.engine
import matlab

eng = matlab.engine.start_matlab()

input_arr = matlab.double(eng.cell2mat([0+1j, -1+1j, -1-1j, 1-1j, 1+0j, 0+0j]), is_complex=True)
result = eng.sc_mapping(input_arr)
print(result)

# ret = eng.triarea(1.0,5.0)
# print(ret)

# ret = eng.linspace(1.0,5.0)
# print(ret)