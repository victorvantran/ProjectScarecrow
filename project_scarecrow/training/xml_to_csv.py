import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET



def xml_to_csv(path):
    fail = 0
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                xmin = int(member.find("bndbox").find("xmin").text)
                ymin = int(member.find("bndbox").find("ymin").text)
                xmax = int(member.find("bndbox").find("xmax").text)
                ymax = int(member.find("bndbox").find("ymax").text)

                width = int(root.find('size').find('width').text)
                height = int(root.find('size').find('height').text)
                    
                value = (root.find('filename').text,
                         int(root.find('size').find('width').text),
                         int(root.find('size').find('height').text),
                         member[0].text,
                         xmin,
                         ymin,
                         xmax,
                         ymax
                         )
                xml_list.append(value)
            except:
                xmin = int(member.find("bndbox").find("xmin").text)
                ymin = int(member.find("bndbox").find("ymin").text)
                xmax = int(member.find("bndbox").find("xmax").text)
                ymax = int(member.find("bndbox").find("ymax").text)
                print(xmin, ymin, xmax, ymax)
                fail = fail + 1
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print(fail)
    return xml_df


def main():
    for folder in ['train','test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


main()
