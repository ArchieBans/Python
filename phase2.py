#Archit Kumar
#V00875990
#June,30'2107

import struct
import sys

def number_to_bytes(number):
    hex_string = format(number, 'x')
    return bytes.fromhex(hex_string)
	
def run_length_decode(finaldecode):
	x = 0
	savenum = 0
	saveindex = 0
	bool = False
	while(x<len(finaldecode)):
		if(finaldecode[x]==128):
			savenum = finaldecode[x+1]
			saveindex = x
			bool = True	
		if(bool):	
			savenum = savenum - 128
			finaldecode.pop(saveindex)
			finaldecode.pop(saveindex)
			for x in range(0,savenum):
				finaldecode.insert(saveindex,129)
			x = 0	
			bool = False
			#print(finaldecode)	
		x = x+1	
	return finaldecode

def run_length_encode(finalencode,convert):
	count = 0
	saveindex = 0
	prev = finalencode[0]
	bool = False
	boolean = False
	x = 0
	literal = 0
	boola = True
	prev = finalencode[0]
	while(x<len(finalencode)):
		curr = finalencode[x]
		if(curr=='\x81' and prev=='\x81'):
			count = count+1
			bool = False
		else:
			bool = True
			literal = count
			count = 0
		if(count==2 and boola):
			saveindex = x - count
			boola = False
			boolean = True
		x = x+1
		prev = curr	
		if(boolean and bool):
			for y in range(0,literal+1):
				finalencode.pop(saveindex)
			zero = 0+(convert-1)
			final2 = number_to_bytes(zero)
			tmep = final2.decode('unicode_escape')
			finalencode.insert(saveindex,tmep)
			literal = literal+(convert-1)
			final2 = number_to_bytes(literal+1)
			tmep = final2.decode('unicode_escape')
			finalencode.insert(saveindex+1,tmep)
			boolean = False
			boola = True	
			count = 0
			x = 0		
	return finalencode

def mtf_decode(infile,outfile):
	file = open(infile,encoding = 'latin-1',mode="r")
	file.seek(4)
	s = file.read(4)
	chars = list(s)
	ints = [ord(c) for c in chars]
	bytes = bytearray(ints)
	result = struct.unpack("I",bytes)
	blocksize = int(result[0])
	file1 = open("testb.ph1",encoding = "latin-1",mode = "w")
	file1.write("\xab\xba\xbe\xef")
	bytes = struct.pack("I",blocksize)
	chars = [chr(c) for c in bytes]
	s = "".join(chars)
	file1.write(s)
	file1.close()
	data = file.read()
	file.close()
	temp = []
	for x in range(0,len(data)):
		temp.insert(x,ord(data[x]))	
	temp = run_length_decode(temp)
	templist = []
	decode = []
	pushed = [] 
	bool = False
	save = 0
	pushed.insert(0,temp[0])	
	templist.insert(0,0)
	for x in range(0,len(temp)):
		if(temp[x]>128):
			fine = temp[x]-128
			for x in range (0,len(pushed)):
				if(fine==pushed[x]):
					save = pushed[x]
					bool = True
			z = len(pushed)
			pushed.insert(z,fine)
			if(bool):
				i = len(decode)
				decode.insert(i,templist[save])
				swap = templist.pop(save)
				templist.insert(1,swap)
				bool = False
		else:
			j = len(templist)
			k = len(decode)
			templist.insert(j,temp[x])
			decode.insert(k,temp[x])	
	fine2 = []
	for x in range(0,len(decode)):
		fine2.insert(x,chr(decode[x]))	
	file1 = open(outfile,encoding = 'latin-1',mode = "a",newline = '')
	for ch in fine2:
		file1.write(ch)
	file1.close()

def mtf_encode(infile,outfile):
	file = open(infile,encoding = 'latin-1',mode = "r")
	file.seek(4)
	s = file.read(4)
	chars = list(s)
	ints = [ord(c) for c in chars]
	bytes = bytearray(ints)
	result = struct.unpack("I",bytes)
	blocksize = int(result[0])
	file1 = open(outfile,encoding = "latin-1",mode = "w")
	file1.write("\xda\xaa\xaa\xad")
	bytes = struct.pack("I",blocksize)
	chars = [chr(c) for c in bytes]
	s = "".join(chars)
	file1.write(s)
	file1.close()
	data = file.read()
	file.close()
	temp = []
	encoding  = []
	ascii = 129
	convert = 129
	size = 0
	boolean = True
	index = 0
	for x in range(1,len(data)):
		if not temp:
			temp.insert(0,data[0])
			final = number_to_bytes(ascii)
			tmep = final.decode('unicode_escape')
			encoding.insert(size,tmep)
			encoding.insert(size+1,data[0])
			size = size+2
			ascii = ascii+1
		for y in range(0,len(temp)):
			if(data[x]==temp[y]):
				index = y+convert
				final2 = number_to_bytes(index)
				tmep = final2.decode('unicode_escape')
				yes = temp.pop(y)
				temp.insert(0,yes)
				encoding.insert(size,tmep)
				size = size+1	
				boolean = False
		if(boolean):
			temp.insert(x,data[x])
			final2 =number_to_bytes(ascii)
			tmep = final2.decode('unicode_escape')
			encoding.insert(size,tmep)
			encoding.insert(size+1,data[x])
			size = size+2
			ascii = ascii+1
			
		boolean = True
	finalencode = list(encoding)
	finalencode = run_length_encode(finalencode,convert)
	yeah = "".join(finalencode)
	file2 = open(outfile,encoding = "latin-1",mode = "a",newline = '')
	file2.write(yeah)
	file2.close()

def main():
	task = ''
	infile = ''
	outfile = ''
	for x in range(0,len(sys.argv)):
		if(sys.argv[x]=='--infile'):
			infile = sys.argv[x+1]
		if(sys.argv[x]=='--outfile'):
			outfile = sys.argv[x+1]
	for x in range(0,len(sys.argv)):
		if(sys.argv[x]=='--encode'):
			mtf_encode(infile,outfile)
		if(sys.argv[x]=='--decode'):
			mtf_decode(infile,outfile)	
if __name__ == '__main__':
	main()