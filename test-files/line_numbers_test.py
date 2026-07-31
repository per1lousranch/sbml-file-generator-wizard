import PySimpleGUI as sg

def repack(widget, option):
    pack_info = widget.pack_info()
    pack_info.update(option)
    widget.pack(**pack_info)

sg.theme('Darkblue')
sg.set_options(font=('Courier New', 20))

column1 = [
    [sg.Multiline('1 ', size=( 5, 10), justification='right', background_color='#202020', expand_x=False, expand_y=False, pad=(0, 0), key='M1', write_only=True,    no_scrollbar=True),
     sg.Multiline('',   size=(40, 10), justification='left',  background_color='#404040', expand_x=False, expand_y=False, pad=(0, 0), key='M2', enable_events=True, focus=True, wrap_lines=False)],
]
layout = [[sg.Column(column1, expand_x=True, expand_y=True, pad=(0, 0), background_color='blue')]]
window = sg.Window('Title', layout, resizable=True, margins=(0, 0), finalize=True)

m1, m2 = window['M1'], window['M2']
repack(m1.widget, {'fill':'y', 'expand':False})
repack(m1.widget.master, {'fill':'y', 'expand':False, 'before':m2.widget.master})
m1.widget.bindtags((str(m1.widget), str(window.TKroot), "all"))
m2.bind('<Configure>', '')
m2.bind('<MouseWheel>', '')
ratio, lines = 0, 1

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'M2 Configure':
        pass
    elif event == 'M2':
        window.refresh()
        new_ratio, _ = m2.vsb.get()
        new_lines = int(m2.widget.index(sg.tk.END).split('.')[0]) - 1
        
        if new_lines != lines:
            lines=new_lines
            text = '\n'.join([f'{i+1} ' for i in range(lines)])
            current = int(m2.widget.index(sg.tk.INSERT).split('.')[0])
            m1.update(text)

        if new_ratio != ratio:
            ratio = new_ratio
            m1.widget.yview_moveto(ratio)

window.close()