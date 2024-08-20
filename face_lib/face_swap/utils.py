# -- coding: utf-8 --
# @Time : 2022/8/26
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

# def reverse2wholeimage_hifi(swaped_img, mat_rev, img_mask, frame_wait_merge, orisize):
#     swaped_img = swaped_img.cpu().numpy().transpose((1, 2, 0))
#     target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
#     img = ne.evaluate('img_mask * (target_image * 255) ')[..., ::-1]
#     img = ne.evaluate('img + frame_wait_merge')
#     final_img = img.astype(np.uint8)
#     return final_img