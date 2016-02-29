#coding=utf-8
'''
    using pygments to highlight the code block contained by <code>...</code>
    in html file
'''
import sys
import re
from pygments import highlight
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.formatters import HtmlFormatter

reload(sys) 
sys.setdefaultencoding('utf8')

def highlight_html(fileName):
    try:
        file = open(fileName,'r')
    except IOError:
        print fileName,"open failed!"
        return

    html = file.read()
    file.close()
    # find out all code_blocks, re.S match over rows
    code_blocks = re.findall(r'<code.*?</code>', html, re.S)

    if len(code_blocks) == 0 :
        print 'code_blocks  not found'
        return
    else:
        print "found", len(code_blocks), "code_blocks!"

    for src_code in code_blocks:
        # find out <code class=''></code> tag and remove them
       	p = re.compile('^<code.*?>|</code>$')
       	src_tag = p.findall(src_code)
       	code = src_code
       	for s in src_tag:
       	    code = code.replace(s, '')

        # find out code language contained in "" or ''
        lang = re.findall("'.*'|\".*\"", src_tag[0])
        # while not given class, guess the lexer by text
        if len(lang) == 0:
            lexer = guess_lexer(code)
        else:
            language = lang[0].strip("'")
            language = language.strip('"')
            #print "colorize in", language
            # when PHP code without the opening and close tag "<?php ?>",
            # using get_lexer_by_name() would not highlight the php code.
            # so a special lexer called PhpLexer() is need, the parameter
            # "startinline=Ture" indicates the php tag is not need 
            if language == 'php':
	        from pygments.lexers import PhpLexer
	        lexer = PhpLexer(startinline=True)
            else:
                lexer = get_lexer_by_name(language)
    	print "colorize in", lexer

        # using pygments to generate highlight_html
        try:
            formatter = HtmlFormatter(linenos='inline', lineostart='0')
            hl_code = highlight(code, lexer, formatter)
            html = html.replace(src_code, hl_code)  # replace with highlight_html
        except:
            print "not hl"
            raise
	outfile = fileName.split(".")[0] + "_hl.html"

    htmlHead = '<!DOCTYPE html>'
    htmlHead += '<html>'
    htmlHead += '<head>'
    htmlHead += '<meta http-equiv="content-type" content="text/html;charset=utf-8" />'
    htmlHead += '<link rel="stylesheet" href="./highlight.css">'
    htmlHead += '</head>'

    html = htmlHead + html
    file = open(outfile, 'w')
    file.write(html)
    file.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "No highlight_html file"
    else:
        for file in sys.argv[1:]:
            highlight_html(file)
    print "enjoy yourself, Bye"
