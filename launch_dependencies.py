import commands, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--instantiation', '-i' ,type=int, default=None, 
                    help='number of instantiation if initialized several times to account for network variability')
global args
args = parser.parse_args()

# submit the first job
cmd = "sbatch launch_epoch1.sh -instantiation %s" % args.instantiation
status, jobnum = commands.getstatusoutput(cmd)
if (status == 0 ):
    print("Job1 is %s" % jobnum)
else:
    print("Error submitting Job1 %s" % jobnum)

for epoch in range(1,500):
    cmd = "sbatch --depend=afterok:%s launch_resume.sh -instantiation %s -epoch %s" % (jobnum, args.instantiation, epoch)
    status,jobnum = commands.getstatusoutput(cmd)
    if (status == 0 ):
        print("Running job is %s" % jobnum)
    else:
        print("Error submitting Job %s" % jobnum)
