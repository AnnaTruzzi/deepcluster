import commands
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--instantiation', '-i' ,type=int, default=None, 
                    help='number of instantiation if initialized several times to account for network variability')
global args
args = parser.parse_args()

# submit the first job
EPOCH_RES = 1
INSTANTIATION = args.instantiation
print('This is instantiation number %s' % INSTANTIATION)
#cmd = "sbatch --export=INSTANTIATION=$%d,EPOCH_RES=$%d launch_resume.sh" % (args.instantiation, epoch)
cmd = "sbatch --export=ALL launch_resume.sh"
status, jobnum = commands.getstatusoutput(cmd)
print(status)
print(jobnum)
if (status == 0 ):
    print("Job1 is %s" % jobnum)
else:
    print("Error submitting Job1 %s" % jobnum)

for EPOCH_RES in range(2,500):
    #cmd = "sbatch --dependency==afterok:%s --export=INSTANTIATION=$%d,EPOCH_RES=$%d launch_resume.sh " % (jobnum, args.instantiation, epoch)
    cmd = "sbatch --dependency==afterok:%s --export=ALL launch_resume.sh " % jobnum
    status,jobnum = commands.getstatusoutput(cmd)
    if (status == 0 ):
        print("Running job is %s" % jobnum)
    else:
        print("Error submitting Job %s" % jobnum)
