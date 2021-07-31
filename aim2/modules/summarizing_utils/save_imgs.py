from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def save_imgs2pdf(img_dir_list, sorting_function, pdf_filename, attr_dict=None):
    img_dir_list_sorted = sorted(img_dir_list, key = sorting_function)
    doc = SimpleDocTemplate(pdf_filename,pagesize=letter,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=18)
    Story=[]
    
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    if attr_dict!=None:
        title = '<font size="25">%s</font>' % "Experiment hyperparameters"
        Story.append(Paragraph(title, styles["Normal"]))
        Story.append(Spacer(1, 50))
        
        for (k, v) in attr_dict.items():
            try:entry = f'{str(k)} : {str(sorted(v, key= float))}'
            except:entry = f'{str(k)} : {str(sorted(v))}'
                
            attr = '<font size="15">%s</font>' % entry
            Story.append(Paragraph(attr, styles["Normal"], bulletText='-'))
            Story.append(Spacer(1, 30))
        
        Story.append(PageBreak())
        

    for idx in range(len(img_dir_list)):
        img_name= img_dir_list_sorted[idx]
        
        img_dir_name= img_name.split('/')[-2]
        ptext = '<font size="7">%s</font>' % img_dir_name
        Story.append(Paragraph(ptext, styles["Normal"])) 
        Story.append(Spacer(1, 10))
        
        ptext = '<font size="12">%s</font>' % img_name
        Story.append(Paragraph(ptext, styles["Normal"])) 
        Story.append(Spacer(1, 30))
        
        im = Image(img_name, 8*inch, 8*inch)
        Story.append(im)
        Story.append(PageBreak())
    doc.build(Story)