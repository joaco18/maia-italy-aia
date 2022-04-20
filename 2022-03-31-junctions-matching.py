import cv2
import utils
import numpy as np

# PARAMETERS
PRUNING_ITERS = 5


def main():
    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'retina_tree.tif'),
        cv2.IMREAD_GRAYSCALE
    )

    # thinning SEs:
    base_north_horizontal_SE = np.array([
        [-1, -1, -1],
        [0,  1,  0],
        [1,  1,  1],
    ])
    base_northeast_diagonal_SE = np.array([
        [0, -1, -1],
        [1,  1, -1],
        [1,  1,  0],
    ])

    thinning_SEs = [
        base_north_horizontal_SE,
        np.rot90(base_north_horizontal_SE, 1),
        np.rot90(base_north_horizontal_SE, 2),
        np.rot90(base_north_horizontal_SE, 3),
        base_northeast_diagonal_SE,
        np.rot90(base_northeast_diagonal_SE, 1),
        np.rot90(base_northeast_diagonal_SE, 2),
        np.rot90(base_northeast_diagonal_SE, 3)
    ]

    current = img.copy()
    previous = np.zeros(img.shape)
    k = 0
    while np.count_nonzero(previous - current):
        print(f'Iteration n: {k}')
        previous = current.copy()
        for SE in thinning_SEs:
            hm_res = cv2.morphologyEx(
                current, cv2.MORPH_HITMISS, SE
            )
            current -= hm_res
        cv2.imshow('Skeletonization in progress', current)
        cv2.waitKey(100)

    cv2.imshow('Skeletonization result', current)
    cv2.waitKey(0)

    skeleton = current.copy()
    cv2.imwrite(
        str(utils.EXAMPLES_DIR / 'retina_tree_skeleton.tif'),
        skeleton
    )

    # Pruning
    base_pruning_SE = np.array([
        [0,  0,  0],
        [-1,  1, -1],
        [-1, -1, -1],
    ])

    pruning_SEs = [
        base_pruning_SE,
        np.rot90(base_pruning_SE, 1),
        np.rot90(base_pruning_SE, 2),
        np.rot90(base_pruning_SE, 3)
    ]

    current = skeleton.copy()
    for k in range(PRUNING_ITERS):
        previous = current.copy()
        for SE in pruning_SEs:
            hm_res = cv2.morphologyEx(
                current, cv2.MORPH_HITMISS, SE
            )
            current -= hm_res
        cv2.imshow('Prunning in progress', current)
        cv2.waitKey(100)
    cv2.imshow('Pruning result', current)
    skeleton_pruned = current.copy()
    cv2.imwrite(
        str(utils.EXAMPLES_DIR / 'retina_tree_pruning.tif'),
        skeleton_pruned
    )

    # Junction SEs
    base_juntions1_SE = np.array([
        [0, -1,  0],
        [1,  1,  1],
        [-1,  1,  0],
    ])
    base_juntions2_SE = np.array([
        [0, -1,  0],
        [1,  1,  1],
        [0,  1, -1],
    ])
    junctions_SEs = [
        base_juntions1_SE,
        np.rot90(base_juntions1_SE, 1),
        np.rot90(base_juntions1_SE, 2),
        np.rot90(base_juntions1_SE, 3),
        base_juntions2_SE,
        np.rot90(base_juntions2_SE, 1),
        np.rot90(base_juntions2_SE, 2),
        np.rot90(base_juntions2_SE, 3)
    ]

    junctions_image = np.zeros(img.shape)
    for SE in junctions_SEs:
        hm_res = cv2.morphologyEx(
            skeleton_pruned, cv2.MORPH_HITMISS, SE
        )
        junctions_image += hm_res

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    junctions_image = cv2.dilate(junctions_image, SE)
    cv2.imshow('Junctions image', junctions_image)
    cv2.waitKey(0)
    junctions_image = np.where(junctions_image != 0, 255, 0)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[junctions_image == 255, :] = (0, 0, 255)

    cv2.imshow('Result', img)
    cv2.waitKey(0)

    cv2.imwrite(
        str(utils.EXAMPLES_DIR/'retina_tree_junctions.tif'),
        img
    )


if __name__ == '__main__':
    main()
