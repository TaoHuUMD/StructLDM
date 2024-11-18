import dominate
from dominate.tags import *

import os
import copy

class HTML:
    def __init__(self, web_dir, title, img_dir = 'images', refresh=0, pre=""):
        self.title = title
        self.web_dir = web_dir

        self.loc_img_dir = img_dir if pre=="" else pre + "_" + img_dir

        self.img_dir = os.path.join(self.web_dir, self.loc_img_dir)
        self.pre = pre

        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            
        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_comp_images(self, ims, txts, links, width=300):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('', link)):
                                img(style="width:%dpx" % width, src=os.path.join('', im))
                            br()
                            p(txt)

    def add_images(self, ims, txts, links, width=512):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join(self.loc_img_dir, link)):
                                img(style="width:%dpx" % (width), src=os.path.join(self.loc_img_dir, im))
                            br()
                            p(txt)

    def save_tmp(self, idx=0):
        tmp_html_file = '%s/tmp_%s_index_%d.html' % (self.web_dir, self.pre, idx)
        f = open(tmp_html_file, 'wt')
        tmp_doc = copy.deepcopy(self.doc)
        f.write(tmp_doc.render())
        f.close()
    
    def save(self):
        html_file = '%s/%s_index.html' % (self.web_dir, self.pre)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.jpg' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.jpg' % n)
    html.add_images(ims, txts, links)
    html.save()
