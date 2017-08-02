import sys
import struct
def rotate(list,n):
		return list[n:]+list[:n]

def chunks(filename,size):
	while True:
		data2 = []
		data = filename.read(size)
		size2 = len(data)
		for i in range(0,size2):
			data2.append(data[i])
		data2.append('\x03')
		if not data:
			break
		yield data2	
def forward(infile,outfile,blocksize):
	fp = open("t05.txt","r")
	file = open("aq.ph1","w")
	file.write('\xab')
	file.write('\xba')
	file.write('\xbe')
	file.write('\xef')
	file.write('\x14')
	file.write('\x00')
	file.write('\x00')
	file.write('\x00')
	file.close()
	for piece in chunks(fp,int(blocksize)):
		temp = []
		tempx  = "foobar"
		tempy = ''
		sorted = []
		sorted.append([])
		size = len(piece)
		sorted.insert(0,piece)
		for x in range(1,size):
			temp = piece
			temp = rotate(temp,x)
			sorted.insert(x,temp)
		sorted.sort()
		temp = []
		for x in range(1,size+1):
			tempx = sorted[x]
			temp.append(tempx[size-1])
		for f in temp:
			tempy = tempy+f
		file = open("aq.ph1","a")
		file.write(tempy)
		file.close()		
def backward(infile,outfile):
	fp = open("t05.ph1","r")
	fp.seek(8)
	for piece in chunks(fp,size=20):
		size = len(piece)
		tempy = [[]for i in range(size)]
		tempz = [[]for i in range(size)]
		l=0
		for x in range(0,size):
			tempz[x].append(piece[x])
		for x in range(0,size-1):
			tempy = list(tempz)
			tempy.sort()
			for x in range(0,size):
				tempz[x].append(tempy[x][l])
			l=l+1
		for x in range(0,size):
			#print(tempz[x])
			if(tempz[x][size-1]=='\x03'):
				found = x
		#print(tempz[found])

if(sys.argv[1]=='--forward'):		
	forward(sys.argv[3],sys.argv[5],int(sys.argv[7]))
else:
	backward(sys.argv[3],sys.argv[5])		