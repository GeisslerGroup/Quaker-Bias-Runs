import numpy as np
import random

zb0 = 0.0
zb1 = 81.0040
n_atoms = 22116
mol_dat = []
# even=False
Rnum_O = 240 * 18
Rnum_Ne = 240 * 18
O_used = 0
Ne_used = 0
data = []

# process a single frame and assign Ne or O to ligands
def analyseFrame(data):
    newdata = np.zeros((n_atoms, 5))
    all_O = False
    O_used = 0
    Ne_used = 0
    all_Ne = False
    in_mol=False
    count = 0

    for j in xrange(len(newdata)):
        if (data[j][0] == '2' and float(data[j][2]) > 0 and not in_mol):
            in_mol=True
            mol_dat = []
            mol_dat.append(data[j])
        elif (data[j][0] == '2' and float(data[j][2]) > 0 and in_mol):
            mol_dat.append(data[j])
        elif (data[j][0] == '1' and float(data[j][2]) > 0):
            if not in_mol:
                raise ValueError("Hit atom type 1 but not in a molecule!")
            mol_dat.append(data[j])
            in_mol=False
            y0 = float(mol_dat[0][2])
            z0 = float(mol_dat[0][3])
            y1 = float(mol_dat[9][2])
            z1 = float(mol_dat[9][3])
#             print y0, y1, z0, z1
            y_vec = y1 - y0
            z_vec = (z1 - z0) / (zb1 - zb0)
            z_vec = (z_vec - round(z_vec)) * (zb1 - zb0)
            th = -np.arctan(z_vec / y_vec)
            if (abs(th) >= 0.6):
                for k in xrange(j-17, j+1, 1):
                    newdata[count][0] = '8'
                    newdata[count][1] = data[k][1]
                    newdata[count][2] = data[k][2]
                    newdata[count][3] = data[k][3]
                    newdata[count][4] = th
                    O_used = O_used + 1
                    count = count + 1
            else:
               for k in xrange(j-17, j+1, 1):
                    newdata[count][0] = '10'
                    newdata[count][1] = data[k][1]
                    newdata[count][2] = data[k][2]
                    newdata[count][3] = data[k][3]
                    newdata[count][4] = th
                    Ne_used = Ne_used + 1
                    count = count + 1
        else:
           newdata[count][0] = data[j][0]
           newdata[count][1] = data[j][1]
           newdata[count][2] = data[j][2]
           newdata[count][3] = data[j][3]
           newdata[count][4] = -100
           count = count + 1

    # find number of remaining bath atoms
    O_left = Rnum_O - O_used
    Ne_left = Rnum_Ne - Ne_used
    return O_left, Ne_left, newdata
#     print O_left, Ne_left, O_used, Ne_used

# write out atoms for that frame
def writeNewFrame(newdata, O_left, Ne_left):
    print "{}\nComment line".format(n_atoms + Rnum_Ne)
#    print newdata

    for j in xrange(n_atoms):
        # start by adding all the oxygens
        if (newdata[j][0] == 8.):
            print "{} {} {} {} {}".format(int(newdata[j][0]), newdata[j][1], newdata[j][2], newdata[j][3], newdata[j][4])
    for k in xrange(O_left):
        print "8 -150 -150 -150 -100"

    for j in xrange(n_atoms):
        # now add all the bath heliums
        if (newdata[j][0] == 10.):
            print "{} {} {} {} {}".format(int(newdata[j][0]), newdata[j][1], newdata[j][2], newdata[j][3], newdata[j][4])
    for k in xrange(Ne_left):
        print "10 -150 -150 -150 -100"

    for j in xrange(n_atoms):
        # now the CH3lig atoms
        if (newdata[j][0] == 2.):
            print "{} {} {} {} {}".format(int(newdata[j][0]), newdata[j][1], newdata[j][2], newdata[j][3], newdata[j][4]) 

    for j in xrange(n_atoms):
        # now the CH2lig atoms
        if (newdata[j][0] == 1.):
            print "{} {} {} {} {}".format(int(newdata[j][0]), newdata[j][1], newdata[j][2], newdata[j][3], newdata[j][4])

    for j in xrange(n_atoms):
        # now the remaining atoms
        if (newdata[j][0] == 3. or newdata[j][0] == 4. or newdata[j][0] == 5. or newdata[j][0] == 6.):
            print "{} {} {} {} {}".format(int(newdata[j][0]), newdata[j][1], newdata[j][2], newdata[j][3], newdata[j][4])

#     if (i == n_atoms * 100):
#         exit(0)

# main run here

with open("dump-0.62.stripped") as f:
    for t,l in enumerate(f):
        l_arr = l.split()
        data.append(l_arr)
        if (t%n_atoms == 0 and t > 0):
            data = np.array(data)
            O_left, Ne_left, newdata = analyseFrame(data)
            writeNewFrame(newdata, O_left, Ne_left)
            data = []

     


# with open("dump.stripped") as f:
#     for t,l in enumerate(f):
# #         if ((t/22116)%2 == 0):
# #             even=True
# #         else:
# #             even=False
# #         print "quicktest {}".format(even)
# 
#         l_arr = l.split()
# #         print(l_arr)
#         l_arr[1] = float(l_arr[1])
#         l_arr[2] = float(l_arr[2])
#         l_arr[3] = float(l_arr[3])
#         if (l_arr[0] == '2' and l_arr[2] > 0 and not in_mol):
#             in_mol=True
# #             mol_dat = []
#             mol_dat.append(l_arr)
#         elif (l_arr[0] == '2' and l_arr[2] > 0 and in_mol):
#             mol_dat.append(l_arr)
#         elif (l_arr[0] == '1' and l_arr[2] > 0):
#             if not in_mol:
#                 raise ValueError("Hit atom type 1 but not in a molecule!")
#             mol_dat.append(l_arr)
#             in_mol=False
#             y0 = mol_dat[0][2]
#             z0 = mol_dat[0][3]
#             y1 = mol_dat[9][2]
#             z1 = mol_dat[9][3]
#             y_vec = y1 - y0
#             z_vec = (z1 - z0) / (zb1 - zb0)
#             z_vec = (z_vec - round(z_vec)) * (zb1 - zb0)
#             th = -np.arctan(z_vec / y_vec)
#             th = random.random()
# #             print y0, z0, y1, z1, y_vec, z_vec, th
#             if (th > 0.5):
#                 for line in mol_dat:
#     # 		    if (line[0]!="8"):print "w to r {}".format(t)
#                     print("O {} {} {}".format(line[1], line[2], line[3]))
#                     O_used = O_used + 1
#             else:
#                 for line in mol_dat:
# # 		    if (line[0]!="2"):print "r to w {}".format(t)
#                     print("Ne {} {} {}".format(line[1], line[2], line[3]))
#                     Ne_used = Ne_used + 1
#             mol_dat=[]
#         else:
#             print(l.strip())
# 
#         if ((t%22116) == 0 and t != 0):
#             O_left = Rnum_O - O_used
#             Ne_left = Rnum_Ne - Ne_used
#             for i in xrange(O_left):
#                 print "O -140 -140 -140"
#             for i in xrange(Ne_left):
#                 print "Ne -140 -140 -140"
#             O_used = 0
#             Ne_used = 0
#             print "{}\nComment line".format(n_atoms + Rnum_O)
# 
#         if t == 22116 * 200:
#             exit(0)

