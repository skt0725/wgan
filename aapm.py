from scipy import io
import matplotlib.pyplot as plt

mat1 = io.loadmat('./data/Low_High_CT_mat_slice/L067/L067_001.mat')
data = mat1.get('imdb')
low = data[0][0][0]
high = data[0][0][1]

fig, ax = plt.subplots(1, 2)
ax[0].imshow(low, cmap="gray")
ax[0].set_title('low_dose')
ax[1].imshow(high, cmap="gray")
ax[1].set_title('high_dose')

plt.savefig('aapm_L067.png')
plt.show()
