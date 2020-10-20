import os
print(os.getcwd())

from subprocess import call
dir = os.getcwd()+"\\tree.dot"
print(dir)
call(['dot','-Tpng',dir,'-o','tree.png','Gdpi=600'])