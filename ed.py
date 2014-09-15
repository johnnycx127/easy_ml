#!/usr/bin/env python
# -*- coding: utf-8 -*-

def ed(s1, s2):
	'''
	>>> ed('eeba', 'abac')
	3
	>>> ed('abc', 'cba')
	2
	>>> ed('cbc', 'eba')
	2
	>>> ed('recoginze', 'recognize')
	1
	>>> ed('sailn', 'failing')
	3
	>>> ed('ab', 'ba')
	1
	'''
	# 动态规划求编辑距离
	# param s1: 字符串1
	# param s2: 字符串2

	len1 = len(s1)
	len2 = len(s2)

	# 初始化矩阵
	matrix = [[i+j for j in range(len2 + 1)] for i in range(len1 + 1)]

	for row in range(len1):
		for col in range(len2):
			comp = [matrix[row+1][col]+1, matrix[row][col+1]+1]

			if s1[row] == s2[col]:
				comp.append(matrix[row][col])
			else:
				comp.append(matrix[row][col]+1)

			# 对相邻字符交换位置的处理判断
			if row > 0 and col > 0:
				if s1[row] == s2[col-1] and s1[row-1] == s2[col]:
					comp.append(matrix[row-1][col-1]+1)

			matrix[row+1][col+1] = min(comp)

	return matrix[len1][len2]

if __name__ == '__main__':
	print ed(u'水杯',u'杯')