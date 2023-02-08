# CS559 Neural Network
# Huiyang Zhao
# UIN 655490960
from sklearn.cluster import KMeans


# put your image generator here
def image_generate(decoder, n=3):
    images = torch.randn(9, 4)
    with torch.no_grad():
        decoded_image = decoder(images.to(device))

    plt.figure()
    for i in range(n):
        # ax = plt.subplot(3, n, i + 1)
        for j in range(n):
            img = decoded_image[i * 3 + j]
            ax = plt.subplot(3, n, j + 1 + i * n)
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('generated_images.pdf')
    plt.show()
    return decoded_image


decoded_image = image_generate(decoder)

# put your clustering accuracy calculation here
kmeans = KMeans(init='k-means++', n_init=10, n_clusters=10, max_iter=60)
kmeans_loader = torch.utils.data.DataLoader(train_data, batch_size=train_data.__len__())
for image_batch, labels in kmeans_loader:
    image_batch = image_batch.to(device)
    kmeans.fit(encoder(image_batch).cpu().detach().numpy())

assignments = []
for i in range(10):
    index_label = np.where(labels == i)
    assignment = [-1]
    for j in range(10):
        index_cluster = np.where(kmeans.labels_ == j)

        assignment.append(len(np.intersect1d(index_label, index_cluster)))
    assignments.append(assignment)

is_matched = [False, False, False, False, False, False, False, False, False, False]
matches = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
matches_row = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

while all(is_matched) is False:
    for i in range(10):
        if matches[i] != -1:
            continue
        max_match = 0
        for j in range(1, 11):
            # print('curr: ' + str(assignments[i][j]) + ' max: ' + str(assignments[i][max_match]))
            if assignments[i][j] > assignments[i][max_match]:
                if is_matched[j - 1] is False:
                    max_match = j - 1
                    matches_row[max_match] = i
                    max_match = j
                else:
                    if assignments[i][j] > assignments[matches_row[j - 1]][j]:
                        is_matched[j - 1] = False
                        matches[matches_row[j - 1]] = -1
                        max_match = j - 1
                        matches_row[max_match] = i
                        max_match = j
        is_matched[max_match - 1] = True
        matches[i] = max_match - 1

total_correct = 0
for index in range(10):
    total_correct = total_correct + assignments[index][matches[index] + 1]
    print(str(index) + ' is mapped to ' + str(matches[index]))

print('Accuracy: ' + '{:.1%}'.format(total_correct / 48000))
