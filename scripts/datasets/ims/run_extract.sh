#SBATCH --job-name=cloud_nowcasting_data_fetch
#SBATCH --error=%x-%j.out
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:01:00

module load anaconda3

python extract_ims_data.py
