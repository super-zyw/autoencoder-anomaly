# f1 = open('train_set.txt', 'w')
# for i in range(800):
#    f1.write('train/ok/' + str(i).zfill(4) + '.bmp' + '\n')
# f1.close()
#
# f2= open('val_set.txt', 'w')
# for i in range(800, 1000):
#     f2.write('train/ok/' + str(i).zfill(4) + '.bmp' + '\n')
# f2.close()

f3= open('test_set.txt', 'w')
for i in range(41):
    f3.write('test/ko/' + str(i).zfill(4) + '.bmp' + '\n')
for i in range(400):
    f3.write('test/ok/' + str(i).zfill(4) + '.bmp' + '\n')

f3.close()