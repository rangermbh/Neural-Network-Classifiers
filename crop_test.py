from PIL import Image
import os
import math

# src_folder = "."
src_folder = "huaxi_crop_test_data"
tar_folder = "tar"
backup_folder = "backup"


def isCrust(pix):
    return sum(pix) < 3


# def hCheck(img, y, step = 50):
def hCheck(img, y, step=10):
    count = 0
    width = img.size[0]
    for x in range(0, width, step):
        if isCrust(img.getpixel((x, y))):
            count += 1
        # if count > width / step / 2:
        if count > width / step / 1.2:
            return True
    return False


def vCheck(img, x, step=10):
    count = 0
    height = img.size[1]
    for y in range(0, height, step):
        if isCrust(img.getpixel((x, y))):
            count += 1
        # if count > height / step / 2:
        # print(height / step /2)
        if count > height / step / 1.2:
            return True
    return False


def boundaryFinder(img, crust_side, core_side, checker):
    if not checker(img, crust_side):
        return crust_side
    if checker(img, core_side):
        return core_side

    mid = (crust_side + core_side) / 2
    while mid != core_side and mid != crust_side:
        if checker(img, mid):
            crust_side = mid
        else:
            core_side = mid
        mid = (crust_side + core_side) / 2
    return core_side
    pass


# mine
"""def boundaryFinder(img, crust_side, core_side, checker):
    the_steps=math.fabs(crust_side-core_side)
    the_steps=the_steps/10
    if not checker(img, crust_side):
        return crust_side
    if checker(img, core_side):
        return core_side

    mid = (crust_side + core_side) / 2
    temp=mid
    i=0
    while mid != core_side and mid != crust_side:
        if checker(img, mid):
            crust_side = mid
        else:
            core_side = mid
        if(not (i==10)):
            i += 1
            if(core_side==0):
                mid=mid-math.fabs(crust_side-core_side)/10
            else:
                mid = mid+ math.fabs(crust_side-core_side) / 10
        else:
            i=0
            mid = (crust_side + core_side) / 2
            print(mid)
    return core_side
    pass
 """


def handleImage(filename, tar, file):
    img = Image.open(os.path.join(src_folder, filename))
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    height = height * 0.89
    # 如果上面有信息，删除个人信息
    left = boundaryFinder(img, 0, width / 2, vCheck)
    right = boundaryFinder(img, width - 1, width / 2, vCheck)
    top = boundaryFinder(img, height * 0.11, height / 2, hCheck)
    bottom = boundaryFinder(img, height - 1, width / 2, hCheck)

    """left = boundaryFinder(img, 0, width/2, vCheck)
    right = boundaryFinder(img, width-1, width/2, vCheck)
    top = boundaryFinder(img, 0, height/2, hCheck)
    bottom = boundaryFinder(img, height-1, width/2, hCheck)"""

    rect = (left, top, right, bottom)
    print(rect)
    region = img.crop(rect)
    print(region)
    """pic_name=os.path.dirname(filename)
    if(not os.path.exists(os.path.join(tar,pic_name))):
        os.makedirs(os.path.join(tar,pic_name))
    region.save(os.path.join(tar,filename),'PNG')"""
    region.save(os.path.join(tar, os.path.basename(filename)), 'PNG')
    pass


def folderCheck(foldername):
    if foldername:
        if not os.path.exists(foldername):
            os.mkdir(foldername)
            print("Info: Folder \"%s\" created" % foldername)
        elif not os.path.isdir(foldername):
            print("Error: Folder \"%s\" conflict" % foldername)
            return False
    return True
    pass


def main():
    if folderCheck(tar_folder) and folderCheck(src_folder) and folderCheck(backup_folder):
        print(src_folder, "src_folfder")
        for filename in os.listdir(src_folder):
            if filename.endswith(".py") or filename.startswith("."): continue
            for filename1 in os.listdir(filename):
                if filename1.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF"):
                    handleImage(os.path.join(filename, filename1), tar_folder, filename)
                    # os.rename(os.path.join(src_folder,os.path.join(filename,filename1)),os.path.join(backup_folder,os.path.join(filename,filename1)))
        pass


if __name__ == '__main__':
    main()
