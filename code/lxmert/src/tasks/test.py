epoch = 1

with open('val_%s.txt' % epoch, 'w') as f:
    q = [[1,2]]
    q = str(q)
    f.write(q)
    f.close()