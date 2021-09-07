from manim_slide import *
import imageio
import math
from scipy.stats import norm
from neural_network_manim import NeuralNetworkMobject



def set_background(self, section, lower_line):
    background = Rectangle(
    width = 16,
    height = 9,
    stroke_width = 0,
    fill_color = WHITE,
    fill_opacity = 1)

    hu_logo = ImageMobject('files/hulogo.png')
    hu_logo.width = 1
    hu_logo.height = 0.5
    emmy = ImageMobject('files/emmy.png')
    emmy.width = 1
    emmy.height = 0.5

    logos = Group(emmy,hu_logo).arrange().move_to(np.array([5, -3.5, 0]))
    
    
      
    line2 = Line(start = np.array([-1.9, -3.5, 0]), 
                 end = np.array([4, -3.5, 0]), stroke_width=0.5).set_color(GRAY)

    pres_title = Text("Uncertainty Quantification for EtE Learning", 
                      font="Noto Sans").scale(0.3).move_to(np.array([-4, -3.5, 0])).set_color(GRAY)

    self.add(background)
    if lower_line == True:
      self.add(pres_title, line2)
    elif lower_line == False:
      self.add(pres_title)
    if section != None: 
        pass
      #line = Line(start = np.array([5, 3, 0]), 
      #            end = np.array([-4, 3, 0]), stroke_width=0.5).set_color(GRAY)
      #section_title = Text(section, font="Noto Sans").scale(0.3).move_to(np.array([-5.5, 3, 0])).set_color(GRAY)
      #self.add(line, section_title)
    self.add(logos)

class Title(SlideScene):
    def construct(self):
        set_background(self, None, True)
        title1 = Tex(r"\fontfamily{lmss}\selectfont \textbf{Marginally Calibrated Predictive Distributions} ").move_to(1*UP).scale(1).set_color(BLUE_E)
        title2 = Tex(r"\fontfamily{lmss}\selectfont \textbf{for End-to-End Learners in Autonomous Driving}").move_to(.4*UP).scale(1).set_color(BLUE_E)

        authors = Tex(r"\fontfamily{lmss}\selectfont Clara Hoffmann and Nadja Klein").scale(0.8).set_color(BLACK).move_to(DOWN)
        group = Tex(r"\fontfamily{lmss}\selectfont Emmy Noether Research Group in Statistics \& Data Science, Humboldt-Universität zu Berlin").scale(0.5).set_color(BLACK).move_to(1.5*DOWN)

        date = Tex(r"\fontfamily{lmss}\selectfont September 16th, 2021").set_color(BLACK).scale(0.6).move_to(2.5*DOWN)
        conference = Tex(r"\fontfamily{lmss}\selectfont Statistische Woche 2021").set_color(BLACK).scale(0.6).move_to(2.75*DOWN)
        self.add(title1, title2,  authors, group, date, conference) # title3,
        self.wait(0.5)

class EtELearning(SlideScene):
    def construct(self):
        set_background(self, "Predictive Densities", True)
        frame_title = Tex(r"\fontfamily{lmss}\selectfont \textbf{End-to-End Learning}").move_to(2.75*UP).set_color(BLACK).scale(0.7)
        ete_diag = ImageMobject('files/ete_diagram/ete.png').move_to(.25*DOWN)
        ete_diag.width = 6
        ete_diag.height = 4
        rectangle_old = Rectangle(height=2, width=2.75, fill_opacity = 0, color = RED).move_to(.5*UP + .3*RIGHT)
        rectangle_old2 = Rectangle(height=1.5, width=1.75, fill_opacity = 0, color = RED).move_to(0.25*UP + 2.4*RIGHT)
        rectangle_new = Rectangle(height=1.5, width=1.5, fill_opacity = 0, color = RED).move_to(1.5*DOWN + 0.17*LEFT)

        self.add(frame_title, ete_diag)
        self.slide_break()
        self.play(Create(rectangle_old))
        self.slide_break()
        self.play(ReplacementTransform(rectangle_old, rectangle_old2))
        self.slide_break()
        self.remove(rectangle_old2)
        self.play(Create(rectangle_new))
        self.wait(0.5)

class UncertaintyEtE(SlideScene):
        
  def show_function_graph(self):
    def func(x, degree):
      #mixture 1
      rv1 = norm(loc = 1.5, scale = 0.4)
      rv2 = norm(loc = 0.5, scale = 0.3)
      rv3 = norm(loc = 0.2, scale = 0.6)
      rv4 = norm(loc = 2, scale = 0.4)
      
      a1 = max([0,5 - degree*2])
      a2 = degree
      a3 = degree
      a4 = 0.5*(degree-1.5)**2

      if x == 0:
        return 0
      
      sum = a1 + a2 + a3 + a4
      return((a1*rv1.pdf(x)/sum + a2*rv2.pdf(x)/sum + a3*rv3.pdf(x)/sum + a4*rv4.pdf(x)/sum )*2)
    
    def func2(x, degree):
      rv1 = norm(loc = 2, scale = 0.7)
      rv2 = norm(loc = 2.4, scale = 0.4)

      if x == 0:
        return 0

      return((.5*rv1.pdf(x) + .5*rv2.pdf(x))*3)
      
      
    p = ValueTracker(0)
    
    ax = Axes(
            (0, 5), (0, 3),
            x_length=5,
            y_length=3,
            axis_config={
                'color' : BLACK,
                'stroke_width' : 1.5,
                'include_numbers' : False,
                'decimal_number_config' : {
                    'num_decimal_places' : 0,
                    'include_sign' : False,
                    'color' : BLACK
                }, 
              },
            tips = False
        ).set_color(BLACK).scale(0.8).move_to(DOWN)

    graph = ax.get_graph(
            lambda x: func(x,  p.get_value()) ,
            color = BLUE 
        )

    graph.add_updater(
            lambda m: m.become(
                ax.get_graph(
                    lambda x: func(x,  p.get_value()),
                    color = BLUE 
                )
            )

        )
    
    label_y = Tex(r"\fontfamily{lmss}\selectfont density").set_color(BLACK).scale(0.5)
    label_x = Tex(r"\fontfamily{lmss}\selectfont steering angle in $^{\circ}$").set_color(BLACK).scale(0.5)
    always(label_y.next_to, ax, LEFT + UP)
    always(label_x.next_to, ax, RIGHT + DOWN)

    label_graph = Tex(r"$\hat{p}(y_i | \boldsymbol{x}_i)$").set_color(BLUE).scale(0.8)
    always(label_graph.next_to, graph, UP)
    
    graph_end = ax.get_graph(
            lambda x: func2(x,  p.get_value()) ,
            color = BLUE 
        )

    return(ax, graph, p, label_y, label_x, graph_end, label_graph)

  def construct(self):

    set_background(self, "Introduction", False) 
    frame_title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Predictive Densities for End-to-End Learning} ").move_to(2.75*UP).set_color(BLACK).scale(0.7)
    theta_tracker = ValueTracker(110)

    line1 = Line(start = 2*LEFT, end = 2*RIGHT, path_arc = np.pi)
    line1.scale(0.9)
    line1.rotate(np.pi)
    line1.move_to(DOWN)
    line1.set_color(BLACK)

    line1b = Line(start = 2*LEFT, end = 2*RIGHT)
    line1b.scale(0.9)
    line1b.rotate(np.pi)
    line1b.move_to(DOWN)
    line1b.set_color(BLACK)


    line2 = DashedLine(start = 2*LEFT, end = 1.2*RIGHT)
    line2.rotate(np.pi/2)
    line2.move_to(LEFT + .25*DOWN)
    line2.set_color(BLACK)

    line3 = DashedLine(start = 2*LEFT, end = 1.2*RIGHT)
    line3.rotate(np.pi/2)
    line3.move_to(RIGHT + .25*DOWN)
    line3.set_color(BLACK)

    line4 = Line(np.array([0,-2,0]), np.array([2,-2,0]))
    line4.set_color(BLACK)
    line_ref = line4.copy()

    rotation_center = np.array([0,-2,0])
    line4.rotate(
            theta_tracker.get_value() * DEGREES, about_point=rotation_center
        )

    self.add(line4, line1, line2, line3) # 
    
    line4.add_updater(
            lambda x: x.become(line_ref.copy()).rotate(
                theta_tracker.get_value() * DEGREES, about_point=rotation_center
            )
        )
    

    roundrect = RoundedRectangle(height = 1, width = 1.75, corner_radius = 0.2)
    roundrect.rotate(angle = np.pi/2)
    roundrect.move_to(3*DOWN)
    roundrect.set_color(BLACK)

    cartop = RoundedRectangle(height = 0.75, width = 0.75, corner_radius = 0.1)
    cartop.rotate(angle = np.pi/2)
    cartop.move_to(3*DOWN)
    cartop.set_color(BLACK)

    windshield =   [(0,0,0),   #P1
                    (0,1,0),    #P2
                    (1,0,0),    #P3
                    (1,1,0),    #P4
                    ]
    

    line5 = Line(np.array([-4,-2.50, 0]), np.array([4, -2.50, 0]), stroke_width = 1.2)
    
    line5.set_color(BLACK)

    bullet_points = Tex(r"\fontfamily{lmss}\selectfont Use predictive densities $\hat{p}(y_i|\boldsymbol{x}_i$) to:").set_color(BLACK).scale(0.7).move_to(RIGHT + 1.25*UP)
    bullet1 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item compare different end-to-end learners \end{itemize} \end{itemize}").set_color(BLACK).scale(0.6).move_to(.75*UP + 1.25*RIGHT)
    bullet2 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item switch to manual control   \end{itemize} \end{itemize}").set_color(BLACK).scale(0.6).move_to(.25*UP).align_to(bullet1, LEFT)
    bullet3 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item improve predictive accuracy  \end{itemize} \end{itemize}").scale(0.6).set_color(BLACK).move_to(-0.25*UP).align_to(bullet1, LEFT) #in different regions
    #bullet4 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item prediction intervals, predictive variance  \end{itemize} \end{itemize}").set_color(BLACK).scale(0.6).move_to(-.75*UP).align_to(bullet1, LEFT)
    #bullet6 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item identify predictive variance  \end{itemize} \end{itemize}").set_color(BLACK).scale(0.7).move_to(-2.5*UP).align_to(bullet1, LEFT)
    #bullet7 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item identify prediction intervals  \end{itemize} \end{itemize}").move_to(DOWN).set_color(BLACK).scale(0.7).move_to(-2.75*UP).align_to(bullet1, LEFT)


    
    self.add(roundrect, cartop, frame_title) #, poly
    self.slide_break()
    self.play(theta_tracker.animate.set_value(90))
    self.play(theta_tracker.animate.set_value(100))
    self.play(theta_tracker.animate.set_value(85))
    self.play(theta_tracker.animate.set_value(95))
    self.slide_break()
    self.play(FadeOut(line2), FadeOut(line3), FadeOut(roundrect), FadeOut(cartop))
    self.remove(line2, line3, roundrect, cartop, line4, line_ref) 
    self.play(ReplacementTransform(line1, line1b)) 
              

    
    #self.remove(line4)
    ax1, graph, p, label_y, label_x, graph_end, label_graph = self.show_function_graph()
    self.play(line1b.animate.move_to(graph.get_bottom()))
  
    self.play(Create(graph), Create(ax1), Create(label_y), Create(label_x), FadeOut(line1b), Create(label_graph))
    self.remove(line5)
    self.wait()
    for i in range(0,5):
      self.play(
              ApplyMethod(p.increment_value,1),
              run_time=1,
          )
      self.wait()
    self.play(ReplacementTransform(graph, graph_end))
    self.slide_break()
    self.remove(label_graph)
    final_graph = VGroup(ax1, label_y, label_x,  graph_end)
    self.remove(ax1, label_y, label_x,  graph_end)
    self.add(final_graph) 
    self.play(final_graph.animate.move_to(4*LEFT).scale(0.5))
    self.slide_break()
    self.play(Create(bullet_points))
    self.slide_break()
    self.play(Create(bullet1))
    self.slide_break()
    self.play(Create(bullet2))
    self.slide_break()
    self.play(Create(bullet3))
    #self.slide_break()
    #self.play(Create(bullet4))
    #self.slide_break()
    #self.play(Create(bullet5))
    self.wait(0.5)

    

class Motivation(SlideScene):
    def construct(self):
        set_background(self, "Motivation", True)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage[T1]{fontenc}")
        myTemplate.add_to_preamble(r"\usepackage{lmodern}")
        
        frame_title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Problems in Uncertainty Quantification for EtE Learning} """).move_to(2.75*UP).set_color(BLACK).scale(0.7)

        
        title_req = Tex(r"\fontfamily{lmss}\selectfont Challenges in EtE Learning").move_to(3*LEFT + 2*UP).scale(0.7).set_color(BLACK)
        bullet1 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item scalability \end{itemize}").scale(0.5).align_to(title_req, LEFT).move_to(3.25*LEFT + 1.5*UP).set_color(BLACK)
        bullet2 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item limited hardware \end{itemize}").scale(0.5).move_to(1*UP).align_to(bullet1, LEFT).set_color(BLACK)
        bullet3 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item real-time prediction \end{itemize}").scale(0.5).move_to(.5*UP).align_to(bullet1, LEFT).set_color(BLACK)
        bullet4 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item complex predictive densities \end{itemize}").scale(0.5).move_to(0*DOWN).align_to(bullet1, LEFT).set_color(BLACK)
        bullet5 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item calibration \end{itemize}").scale(0.5).move_to(0.5*DOWN).align_to(bullet1, LEFT).set_color(BLACK)
        problems = VGroup(title_req, bullet1, bullet2, bullet3, bullet4, bullet5)
        title = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item  Shortcomings of current methods for uncertainty quantification: \end{itemize}").move_to(3*RIGHT + 2*UP).set_color(BLACK).scale(0.5)
        problems1 = Tex(r""" \fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item[$\rightarrow$] Bayesian Neural Networks (BNNs) \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).align_to(title, LEFT).move_to(1.5*UP + 2*RIGHT)
        problems2 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item[$\rightarrow$] MC-dropout, ensembles  \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).move_to(.75*UP).align_to(problems1, LEFT)
        problems3 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item[$\rightarrow$] neural linear models (NLMs) \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).move_to(0*DOWN).align_to(problems1, LEFT)
        problems4 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item[$\rightarrow$]  usually no guarantees \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).move_to(0.5*DOWN).align_to(problems1, LEFT)

        solution = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Solution}: Implicit-copula neural linear model of Klein, Nott, Smith (2021) """).set_color(BLACK).scale(.7).align_to(title, LEFT).move_to(2*DOWN)
        solution.bg = SurroundingRectangle(solution, color=GREEN_D, fill_color=GREEN_A, fill_opacity=.2)
        citation = Tex(r"\fontfamily{lmss}\selectfont Klein, N., Nott, D. J. and Smith, M. S. (2021). Marginally calibrated deep distributional regression. \textit{Journal of Computational and Graphical Statistics}").scale(0.4).move_to(2.9*DOWN + 2.75*LEFT).set_color(BLACK)
        
        solutions = VGroup(solution, solution.bg)

        self.add(frame_title) #, title
        self.add(problems)
        self.slide_break()

        self.slide_break()
        self.play(Create(problems1))
        self.slide_break()
        self.play(Create(problems2))
        self.slide_break()
        self.play(Create(problems3))
        self.slide_break()
        self.play(Create(problems4))
        self.slide_break()
        self.play(Create(solutions))
        self.add(citation)
        self.wait(0.5)

class Notation(SlideScene):
    def construct(self):
        set_background(self, "Variables and Notation", True)

        title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Predictive Densities in EtE Learning} """).move_to(2.75*UP).set_color(BLACK).scale(0.9)

        input_im = ImageMobject('files/angle_20_small.png').scale(0.3).move_to(4*LEFT + 1*UP )
        input_test = Tex(r'\fontfamily{lmss}\selectfont input image $\boldsymbol{x}_i$').scale(0.5).set_color(BLACK).move_to(input_im.get_bottom() + .25*DOWN)
        
        #st_wheel = ImageMobject('files/steering_wheel.jpeg').scale(0.1).move_to(1*UP)
        #st_text = Tex(r'\fontfamily{lmss}\selectfont steering angle $y_i$').move_to(st_wheel.get_bottom() + .5*DOWN).scale(0.5).set_color(BLACK)
        dnn = Tex(r"""\fontfamily{lmss}\selectfont DNN""").set_color(BLACK).move_to(UP)
        dens_im =  ImageMobject('files/angle_20.png').scale(0.9).move_to(4*RIGHT + 1*UP)
        pred = Tex(r'\fontfamily{lmss}\selectfont predictive density $p(y_i | \boldsymbol{x}_i)$').move_to(dens_im.get_bottom() + .25*DOWN).scale(0.5).set_color(BLACK)
        
        arrow1 = Arrow(input_im.get_right()  , dnn.get_left() , 
                            buff=0, stroke_width= 5, 
                            max_tip_length_to_length_ratio = 0.05).set_color(BLACK).scale(0.2)
        arrow2 = Arrow(dnn.get_right()  , dens_im.get_left() , 
                            buff=0, stroke_width= 5, 
                            max_tip_length_to_length_ratio = 0.05).set_color(BLACK).scale(0.2)

        bullets_0 = Tex(r"""\fontfamily{lmss}\selectfont Problems of current methods to estimate $p(y_i | \boldsymbol{x}_i)$""").move_to(1*DOWN + 3*LEFT ).set_color(BLACK).scale(0.5)
        bullets = Tex(r"""\fontfamily{lmss}\selectfont
        \begin{itemize} 
        \item not scalable (Bayesian Neural Networks)
        \item not calibrated (Mixture Density Networks)
        \item have to evaluate many DNNs in parallel (MC-dropout, ensembles)
         \end{itemize}""").scale(0.5).move_to(bullets_0.get_bottom()+ 1*DOWN).set_color(BLACK).align_to(bullets_0, LEFT)

        solution = Tex(r""" Marginally calibrated deep distributional Klein, Nott, Smith (2021) $=$ Implicit-copula neural linear model""").move_to(bullets.get_bottom() + .25*DOWN).align_to(bullets_0, LEFT)

        self.add(title, input_im, input_test,  pred, dens_im, bullets, bullets_0, arrow1, arrow2, dnn) #st_wheel, st_text,
        self.wait(0.5)

class NLM(SlideScene):
  def construct(self):
    set_background(self, "Implicit-Copula Neural Linear Model", True)

    title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Implicit-Copula Neural Linear Model} """).move_to(2.75*UP).set_color(BLACK).scale(0.9)


    input = ImageMobject('files/80_436run1.png').scale(0.05).move_to(3*LEFT + .5*DOWN)
    input_test = Tex(r'input image $\boldsymbol{x}_i$').move_to(input.get_bottom()).scale(0.5).set_color(BLACK)

    myNetwork = NeuralNetworkMobject([8, 5, 1]).set_color(LIGHT_GRAY).scale(0.75).move_to(.5*DOWN)
    output_text1 = Tex(r"""\fontfamily{lmss}\selectfont predicted""").move_to(myNetwork.get_right() + 1*RIGHT + .5*UP).scale(0.6).set_color(BLACK) #.move_to(1.7*RIGHT)
    output_text2 = Tex(r' \fontfamily{lmss}\selectfont steering angle').move_to(output_text1.get_bottom() + 0.3*DOWN).scale(0.6).set_color(BLACK)
    output_text3a = Tex(r"$\hat{y}_i$",
                        r"$=$",
                        r"$\psi_{\boldsymbol{\zeta}}(\boldsymbol{x}_i)$",
                        r"$\boldsymbol{\beta}$").move_to(output_text2.get_bottom()+ 0.3*DOWN).scale(0.6).set_color(BLACK)
    output_text3a.set_color_by_tex_to_color_map({
        r"$\psi_{\boldsymbol{\zeta}}(\boldsymbol{x}_i)$": RED_D,
        r"$\boldsymbol{\beta}$": GREEN_D,
             }) 
    output_text3b = Tex(r"$\hat{z}_i$",
                        r"$=$",
                        r"$\psi_{\boldsymbol{\zeta}}(\boldsymbol{x}_i)$",
                        r"$\boldsymbol{\beta}$").set_color(BLACK).move_to(output_text2.get_bottom()+ 0.3*DOWN).scale(0.6)
    output_text3b.set_color_by_tex_to_color_map({
        r"$\psi_{\boldsymbol{\zeta}}(\boldsymbol{x}_i)$": RED_D,
        r"$\hat{z}_i$": PURPLE_D,
        r"$\boldsymbol{\beta}$": GREEN_D
             }) 
    output_text4 = Tex(r'$\boldsymbol{\beta} | \boldsymbol{\theta}, \sigma^2 \sim N(0, \sigma^2P(\boldsymbol{\theta})^{-1})$').move_to(output_text3a.get_bottom()+ 0.3*DOWN).scale(0.6).set_color(GREEN_E).align_to(output_text3a, LEFT)
    output_text5 = Tex(r"$z_i = \Phi^{-1}F_Y(y_i)$").move_to(output_text4.get_bottom()+ 0.3*DOWN).scale(0.6).set_color(PURPLE_D).align_to(output_text4, LEFT)
    Bzeta = Rectangle(
        width = .6,
        height = 2.75,
        stroke_width = 2,
        color = RED,
        fill_color = GRAY,
        fill_opacity = 0).move_to(myNetwork.get_center())

    Bzeta_text = Tex(r"""\fontfamily{lmss}\selectfont deep basis functions $\psi_{\boldsymbol{\zeta}}(\boldsymbol{x_i})$""").move_to(Bzeta.get_bottom() + .5*DOWN).set_color(RED_D).scale(0.7)

    add_error = Tex(r""" $+ \varepsilon_i$,   $ \quad \varepsilon_i \sim N(0,\sigma^2)$""").move_to(output_text3a.get_right() + 1.4*RIGHT).scale(0.6).set_color(BLACK)
    cnn = Rectangle(
              width = 1,
              height = 1,
              stroke_width = 1,
              fill_color = GRAY,
              fill_opacity = 1).move_to(3*LEFT  + .5*DOWN)

    cnn1 = cnn.copy().move_to(cnn.get_center() + np.array([.25,.25,0])).set_color(GRAY)
    cnn2 = cnn.copy().move_to(cnn.get_center() + np.array([.5,.5,0])).set_color(GRAY)
    
    cnn_text = Text('Conv Layers').move_to(cnn.get_center()).scale(0.2).set_color(BLACK)
    cnn_all = VGroup (cnn2, cnn1, cnn, cnn_text)

    # Arrows
    arrow_input_cnn = Arrow(input.get_right()  , myNetwork.get_left() , 
                            buff=0, stroke_width= 2, 
                            max_tip_length_to_length_ratio = 0.05).set_color(BLACK).scale(0.2)
    arrow_cnn_dnn = Arrow(cnn.get_right() + 0.25*RIGHT, myNetwork.get_left() + 0.5*RIGHT, buff=0, 
                            stroke_width= 2, 
                            max_tip_length_to_length_ratio = 0.05).set_color(BLACK).scale(0.2)
    
    

    #myNetwork.label_outputs_text([r'predicted steering angle'])
    citation = Tex(r"\fontfamily{lmss}\selectfont Klein, N., Nott, D. J. and Smith, M. S. (2021). Marginally calibrated deep distributional regression. \textit{Journal of Computational and Graphical Statistics}").scale(0.4).move_to(2.75*DOWN + 2.75*LEFT).set_color(BLACK)
        

    self.add(title)
    self.add(input, input_test, citation)
    self.play(Create(arrow_input_cnn))
    self.slide_break()
    #self.play(Create(cnn_all))
    #self.slide_break()
    #self.play(Create(arrow_cnn_dnn))
    self.slide_break()
    self.play(Write(myNetwork))
    self.slide_break()
    self.play(Create(Bzeta), Create(Bzeta_text))
    self.slide_break()
    self.play(Create(output_text1), Create(output_text2)) # 
    self.slide_break()
    self.play(Create(output_text3a))
    self.slide_break()
    self.play(Create(add_error))
    self.slide_break()
    self.play(Create(output_text4))
    self.slide_break()
    self.play(ReplacementTransform(output_text3a, output_text3b))
    self.slide_break()
    self.play(Create(output_text5))
    self.wait(0.5)

class CopulaSlide(SlideScene):
  def construct(self):
    set_background(self, "Copula Model", True)


    title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Implicit Copula Neural Linear Model (IC-NLM)} """).move_to(2.75*UP).set_color(BLACK).scale(0.9)

    nlm_text = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item pseudo regression: \end{itemize}").set_color(BLACK).scale(0.9)
    nlm1 = Tex(r"$\Tilde{Z} = $").set_color(BLACK).scale(0.9) #.move_to(RIGHT)
    nlm2 = Tex(r"$\psi_{\boldsymbol{\zeta}}(\boldsymbol{x})\boldsymbol{\beta}$").set_color(BLACK).scale(0.9)
    nlm3 = Tex(r"$ + \boldsymbol{\varepsilon}$").set_color(BLACK).scale(0.9)
    nlm = VGroup(VGroup(nlm_text, nlm1).arrange(RIGHT, buff=MED_SMALL_BUFF), nlm2, nlm3).arrange(RIGHT, buff=SMALL_BUFF).move_to(np.array([-1.5, 1, 0])).scale(0.7)

    nlm_text_dbf_text = Tex(r"\fontfamily{lmss}\selectfont deep basis functions").move_to(nlm2.get_bottom() + DOWN).set_color(BLACK).scale(0.6)
    nlm_text_dbf_line =  Line(nlm2.get_bottom(), nlm_text_dbf_text.get_top(), stroke_width = 1).set_color(BLACK)
    nlm_text_dbf = VGroup(nlm_text_dbf_text, nlm_text_dbf_line)
    
    nlm_text_error_text = Tex(r"\fontfamily{lmss}\selectfont artificial error term").move_to(nlm3.get_bottom()+ 2*DOWN + 0.5*RIGHT).set_color(BLACK).scale(0.6)
    nlm_text_error_line =  Line(nlm3.get_bottom() + 0.05*RIGHT ,nlm_text_error_text.get_top(), stroke_width = 0.5).set_color(BLACK)
    nlm_text_error = VGroup(nlm_text_error_text, nlm_text_error_line)


    errors = Tex(r"$\varepsilon_i \sim N(0, \sigma^2)$").set_color(BLACK).move_to(nlm.get_right()+ 1.5*RIGHT).scale(0.9).scale(0.7)
    errors2_text_text = Tex(r"\fontfamily{lmss}\selectfont i.i.d.").move_to(errors.get_bottom()+ DOWN).set_color(BLACK).scale(0.6)
    errors2_line =  Line(errors.get_bottom(),errors2_text_text.get_top(), stroke_width = 0.5).set_color(BLACK)
    errors2_text = VGroup(errors2_text_text, errors2_line)


    beta_text = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item  shrinkage prior: \end{itemize} ").move_to(RIGHT).set_color(BLACK).move_to(errors.get_bottom() + .5*DOWN).scale(0.9)
    beta1 = Tex(r" \quad $\boldsymbol{\beta} | \boldsymbol{\theta}, \sigma^2 \sim N(0, \sigma^2$").set_color(BLACK).scale(0.9)
    beta12 = Tex(r"$P(\boldsymbol{\theta})^{-1})$").set_color(BLACK).scale(0.9)
    beta = VGroup(VGroup(beta_text, beta1).arrange(RIGHT, buff=MED_SMALL_BUFF), beta12).arrange(RIGHT, buff=SMALL_BUFF).scale(0.7).move_to(nlm.get_bottom() + 1*DOWN).align_to(nlm, LEFT)

    beta1_text_text = Tex(r"\fontfamily{lmss}\selectfont shrinkage parameters").move_to(beta12.get_bottom() + DOWN).set_color(BLACK).scale(0.6)
    beta1_line =  Line(beta12.get_bottom(), beta1_text_text.get_top(), stroke_width = 1).set_color(BLACK)
    beta_text = VGroup(beta1_text_text, beta1_line)

    z_text = Tex(r" \fontfamily{lmss}\selectfont \begin{itemize} \item standardized distribution: \end{itemize}").set_color(BLACK).move_to(beta.get_bottom() + .5*DOWN).scale(0.9)
    z_tex = Tex(r"$ Z = \sigma^{-1}S(\boldsymbol{x}, \boldsymbol{\theta}) \Tilde{Z} | \boldsymbol{x}, \sigma^2 \sim N(0, R(\boldsymbol{x}, \boldsymbol{\theta}))$").set_color(BLACK).scale(0.9)
    z = VGroup(z_text, z_tex).arrange(RIGHT, buff=SMALL_BUFF).scale(0.7).move_to(beta.get_bottom() + 1*DOWN).align_to(nlm, LEFT)

    z_text_text = VGroup(Tex(r"$\boldsymbol{\beta}$"), Tex(r"\fontfamily{lmss}\selectfont integrated out")).arrange(RIGHT).move_to(z.get_bottom() + DOWN).set_color(BLACK).scale(0.6)
    z_text_line =  Line(z.get_bottom(), z_text_text.get_top(), stroke_width = 1).set_color(BLACK)
    z_text = VGroup(z_text_text, z_text_line)

    z_text_text2 = VGroup(Tex(r"\fontfamily{lmss}\selectfont margins"), Tex(r"$\sim N(0,1)$")).arrange(RIGHT).move_to(z.get_bottom() + DOWN + 4*RIGHT).set_color(BLACK).scale(0.6)
    z_text_line2 =  Line(z.get_bottom() + 4*RIGHT, z_text_text2.get_top(), stroke_width = 1).set_color(BLACK)
    z_text2 = VGroup(z_text_text2, z_text_line2)


    rectangle = Rectangle(height=2, width=3)

    #sklars = Text(r"Sklar's theorem", font="Noto Sans").set_color(BLACK).scale(0.5).move_to(.5*UP)
    sklars = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Sklar's Theorem} """).move_to(2.5*UP).set_color(BLACK).scale(0.7)
    n = 2
    sklar_eq = Tex(r"$p(\boldsymbol{z} | \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$ = $",
                   r"$c_{DNN}$",
                   r"$($"
                   r"$F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})$",
                   r"$| \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$\prod_{i=1}^n$",
                   r"$p_{z_i}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$"
                      ).move_to(sklars.get_bottom() + n*DOWN).set_color(BLACK).scale(0.6)
    
    sklar_eq_col = sklar_eq.copy()
    sklar_eq_col.set_color_by_tex_to_color_map({
        r"$p(\boldsymbol{z} | \boldsymbol{x}, \boldsymbol{\theta})$": RED_E,
        r"$p_{z_i}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$": RED_E,
        r"$F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})$" :  RED_E,
        r"$c_{DNN}$" : BLUE
             }) 

    sklar_eq2 = Tex(r"$\phi_n(\boldsymbol{z}; \boldsymbol{0}, R(\boldsymbol{x}, \boldsymbol{\theta}))$",
                   r"$ = $",
                   r"$c_{DNN}$",
                   r"$($",
                   r"$F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})$",
                   r"$| \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$\prod_{i=1}^n $",
                   r"$\phi_{1}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$"
                      ).move_to(sklars.get_bottom() + n*DOWN).set_color(BLACK).scale(0.6)
    sklar_eq2.set_color_by_tex_to_color_map({
        r"$\phi_n(\boldsymbol{z}; \boldsymbol{0}, R(\boldsymbol{x}, \boldsymbol{\theta}))$": RED_E,
        r"$F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})$" :  RED_E,
        r"$\phi_{1}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$": RED_E
             })   
    
    # should be nicer to move...
    copula_dens = Tex(r"$c_{DNN}(F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})| \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$ = $",
                   r"$\frac{\phi_n(\boldsymbol{z}; \boldsymbol{0}, R(\boldsymbol{x}, \boldsymbol{\theta})}{\prod_{i=1}^n \phi_{1}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})}$"
                      ).move_to(sklars.get_bottom() + n*DOWN).set_color(BLACK).scale(0.6)    
    copula_dens.set_color_by_tex_to_color_map({
        r"$\frac{\phi_n(\boldsymbol{z}; \boldsymbol{0}, R(\boldsymbol{x}, \boldsymbol{\theta})}{\prod_{i=1}^n \phi_{1}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})}$" : RED_E
        })   

    
    # Now we use the same copula to model y with the shrinkage parameters integrated out
    # note that even though the copula from before is a Gaussian copula
    # the copula with the shrinkage parameters integrated out can be far from Gaussian
    copula_dens_y = Tex(r"$p(\boldsymbol{y}| \boldsymbol{x}, \boldsymbol{\theta}) = c_{DNN} ($",
                        r"$F_Y(y_1), \ldots, F_Y(y_n)$",
                         r"$| \boldsymbol{x}, \boldsymbol{\theta}) \prod_{i=1}$",
                         r"$F_Y(y_i)$"
                   ).move_to(copula_dens.get_bottom() + n/2*DOWN).set_color(BLACK).scale(0.6)
    
    copula_dens_y.set_color_by_tex_to_color_map({
        r"$F_Y(y_1), \ldots, F_Y(y_n)$": RED_E,
        r"$F_Y(y_i)$": RED_E
             })
    
    # this copula density involves inverting an nxn matrix 
    # as a solution we evaluate the density of y conditional on the model parameters
    # and integrate over them, so that the final expression we're interest in is
    final_exp_title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Target Expression} """).move_to(2.5*UP).set_color(BLACK).scale(0.7)
    n = 2
    final_expr = Tex(r"$p(\boldsymbol{y} | \boldsymbol{x}) = \int p(\boldsymbol{y}| \boldsymbol{x}, \boldsymbol{\beta}, \boldsymbol{\theta}) $",
                     r"$p(\boldsymbol{\beta}, \boldsymbol{\theta}| \boldsymbol{x}, \boldsymbol{y})$",
                     r"$d(\boldsymbol{\beta}, \boldsymbol{y})$").set_color(BLACK).scale(0.8)
    
    final_expr.set_color_by_tex_to_color_map({
        r"$p(\boldsymbol{\beta}, \boldsymbol{\theta}| \boldsymbol{x}, \boldsymbol{y})$": RED_E
             })
    final_exp_text_text = Tex(r"\fontfamily{lmss}\selectfont known up to normalizing constant").move_to(final_expr.get_bottom() + DOWN + RIGHT ).set_color(BLACK).scale(0.6)
    final_exp_text_line =  Line(final_expr.get_bottom() + RIGHT, final_exp_text_text.get_top(), stroke_width = 1).set_color(BLACK)
    final_exp_text = VGroup(final_exp_text_text, final_exp_text_line)
    

    # the posterior of the model parameters is only known up to a normalizing constant so it has to be approximated.
    # in former work, the posterior was estimated in MCMC
    # However, this is not possible in EtE learning, where we have immense data sets
    # With the data set sizes typical in EtE learning, MCMC samplers become slow or sometimes even don't converge at all

    # To ensure fast and scalable estimation we employ a variational inference approach

    # REGRESSION MODEL
    
    #self.add(title, nlm)
    #self.slide_break() 
    #self.play(Create(nlm_text_dbf))
    #self.slide_break() 
    #self.play(Create(nlm_text_error))
    #self.slide_break() 
    #self.remove(nlm_text_dbf, nlm_text_error)
    #self.slide_break() 

    # ERROR TERM
    #self.play(Create(errors))
    #self.slide_break() 
    #self.play(Create(errors2_text))
    #self.slide_break() 
    #self.remove(errors2_text)
    #self.slide_break() 
    

    # BETA
    #self.play(Create(beta))
    #self.slide_break() 
    #self.play(Create(beta_text))
    #self.slide_break()
    #self.remove(beta_text)
    #self.slide_break()

    # beta integrated out
    #self.play(Create(z))
    #self.slide_break()
    #self.play(Create(z_text))
    #self.slide_break() 
    #self.play(Create(z_text2))
    #self.slide_break() 
    #self.remove(z_text, z_text2)
    #self.slide_break() 

    #self.remove(nlm, errors, beta,  z, z_text2, title) #beta2,
    #self.slide_break() 

    # SKLARS THEOREM
    self.add(sklars, sklar_eq)
    self.slide_break() 
    
    self.play(ReplacementTransform(sklar_eq, sklar_eq_col))
    self.slide_break() 
    self.play(ReplacementTransform(sklar_eq_col, sklar_eq2))
    self.slide_break() 
    self.play(ReplacementTransform(sklar_eq2, copula_dens))
    self.slide_break() 
    self.add(copula_dens_y)
    self.slide_break() 

    self.remove(copula_dens, copula_dens_y, sklars)
    self.play(ReplacementTransform(copula_dens_y, final_expr))
    self.add(final_exp_title)
    self.slide_break() 
    self.add(final_exp_text)
    self.slide_break(0.5) 




class VIvsHMC(SlideScene):
    def construct(self):
        set_background(self, "VI vs. HMC", True)
        title = Tex(r"""\fontfamily{lmss}\selectfont 
        \textbf{Two Approaches to estimate}  $p(\boldsymbol{\beta}, \boldsymbol{\theta}|\boldsymbol{x}, \boldsymbol{y})$""").move_to(2.75*UP).set_color(BLACK).scale(0.9)
        mcmc = Tex(r"""\fontfamily{lmss}\selectfont Markov chain Monte-Carlo""").move_to(3.5*LEFT + 1.25*UP).set_color(BLACK).scale(0.6)
        vi = Tex(r"""\fontfamily{lmss}\selectfont Variational Inference""").move_to(3.5*RIGHT + 1.25*UP).set_color(BLACK).scale(0.6)
        
        p = ValueTracker(1000)

        text, number = label = VGroup(
            Tex(r"""\fontfamily{lmss}\selectfont scalability for $n = $""").set_color(BLACK),
            DecimalNumber(
                0,
                show_ellipsis=True,
                num_decimal_places=1,
                include_sign=False,
            ).set_color(BLACK)
        )
        label.arrange(RIGHT).move_to(.5*UP).scale(0.5)
        f_always(number.set_value, p.get_value)

        scal_mcmc = Tex(r"""\fontfamily{lmss}\selectfont sticky, slow \xmark""").set_color(RED_D).move_to(3.5*LEFT + .5*UP).scale(0.6)
        scal_vi = Tex(r"""\fontfamily{lmss}\selectfont scalable \checkmark""").set_color(GREEN_D).move_to(3.5*RIGHT + .5*UP).scale(0.6)
        
        acc_type = Tex(r"\fontfamily{lmss}\selectfont accuracy").move_to(-.5*UP).scale(0.5).set_color(BLACK)
        acc_mcmc = Tex(r"""\fontfamily{lmss}\selectfont exact sampling \checkmark """).set_color(GREEN_D).move_to(3.5*LEFT + -.5*UP).scale(0.6)
        acc_vi = Tex(r"""\fontfamily{lmss}\selectfont approximative \xmark""").set_color(RED_D).move_to(3.5*RIGHT + -.5*UP).scale(0.6)

        estimation_type = Tex(r"\fontfamily{lmss}\selectfont estimation type").move_to(-1.5*UP).scale(0.5).set_color(BLACK)
        prop_mcmc = Tex(r"""\fontfamily{lmss}\selectfont numerical sample""").set_color(BLACK).move_to(3.5*LEFT + -1.5*UP).scale(0.6).set_color(GREEN_D)
        prop_vi = Tex(r"""\fontfamily{lmss}\selectfont closed-form solution """).set_color(BLACK).move_to(3.5*RIGHT + -1.5*UP).scale(0.6).set_color(GREEN_D)

        self.add(title)
        self.add(mcmc)
        self.add(vi)
        self.slide_break()
        self.add(label)
        self.play(p.animate.set_value(1000000), run_time=7)
        self.play(Create(scal_mcmc), Create(scal_vi))

        self.slide_break()
        self.add(acc_type)
        self.slide_break()
        self.play(Create(acc_mcmc), Create(acc_vi))
        self.slide_break()
        self.add(estimation_type)
        self.slide_break()
        self.play(Create(prop_mcmc), Create(prop_vi))
        self.wait(0.5)


class VISlide(SlideScene):
  def construct(self):
    set_background(self, "Variational Inference", True)
    title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Variational Inference}""").move_to(2.75*UP).set_color(BLACK).scale(0.9)
    var_density = Tex(r"""$q_{\boldsymbol{\lambda}}(\boldsymbol{\beta}, \boldsymbol{\theta})$""").set_color(RED_E)
    member = VGroup(Text("approximation family:", font="Noto Sans").set_color(BLACK).scale(0.5),
                    Tex(r"""$q_{\boldsymbol{\lambda}}(\boldsymbol{\beta}, \boldsymbol{\theta}) = 
    N(\boldsymbol{\mu}, \boldsymbol{\Upsilon} \boldsymbol{\Upsilon}^T + \Delta^2)$""").set_color(BLACK)).arrange(DOWN).move_to(2*RIGHT +1.5*UP ).scale(0.6)


    KLD = VGroup(Text("Kullback-Leibler divergence:", font="Noto Sans").set_color(BLACK).scale(0.5),
                 Tex(r"""$\mathrm{KLD} (q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta}) || p(\boldsymbol{\vartheta}| \boldsymbol{y})) = 
    \int \frac{q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})}{p(\boldsymbol{\vartheta} | \boldsymbol{y})}
    q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})d(\boldsymbol{\vartheta})$""").set_color(BLACK)).arrange(DOWN).move_to(.5*UP + 2*RIGHT ).scale(0.6)

    vlb = VGroup(Text("variational lower bound:", font="Noto Sans").set_color(BLACK).scale(0.5),
                 Tex(r"""$\mathcal{L}(\boldsymbol{\lambda}) = 
    \mathbb{E}_{q_{\boldsymbol{\lambda}}} [\log(p(\boldsymbol{y}|\boldsymbol{\vartheta})p(\boldsymbol{\vartheta})) - 
    \log q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})] $""").set_color(BLACK)).arrange(DOWN).move_to(.5*UP + 2*RIGHT ).scale(0.6)
    
    delta_vlb = VGroup(Text("optimize via gradient:", font="Noto Sans").set_color(BLACK).scale(0.5),
                       Tex(r"$\nabla_{\boldsymbol{\lambda}}\mathcal{L}(\boldsymbol{\lambda})$").set_color(BLACK)).arrange(DOWN).move_to(2*RIGHT + 1*DOWN).scale(0.6)
    
    delta_vlb_update =  VGroup(Text("update rule:", font="Noto Sans").set_color(BLACK).scale(0.5),
                               Tex(r"""$ \boldsymbol{\lambda}^{(t+1)} = 
    \boldsymbol{\lambda}^{(t)} + \boldsymbol{\rho}^{(t)} 
    \nabla_{\boldsymbol{\lambda}}\mathcal{L}(\boldsymbol{\lambda}^{(t)})$""").set_color(BLACK)).arrange(DOWN).move_to(2*RIGHT +2*DOWN).scale(0.6)


    axes = Axes((0, 3), (0, 5)).scale(0.4).move_to(.5*LEFT)


    def func(x):
      #mixture 1
      rv1 = norm(loc = 1.5, scale = 0.2)
      rv2 = norm(loc = 0.5, scale = 0.5)
      rv3 = norm(loc = 0, scale = 0.4)
      rv4 = norm(loc = 2.2, scale = 0.3)
      
      a1 = 1
      a2 = 1
      a3 = 1
      a4 = 1

      if x == 0:
        return 0
      
      sum = a1 + a2 + a3 + a4
      return((a1*rv1.pdf(x)/sum + a2*rv2.pdf(x)/sum + a3*rv3.pdf(x)/sum + a4*rv4.pdf(x)/sum )*3)
      
    def func2(x, degree):
      #mixture 1
      rv1 = norm(loc = 1.5, scale = 0.2)
      rv2 = norm(loc = 0.5, scale = 0.5)
      rv3 = norm(loc = 0, scale = 0.4)
      rv4 = norm(loc = 2.2, scale = 0.3)
      
      a1 = 5-degree
      a2 = 5-degree
      a3 = 3-degree/2
      a4 = 9-degree*2

      if x == 0:
        return 0
      
      sum = a1 + a2 + a3 + a4
      return((a1*rv1.pdf(x)/sum + a2*rv2.pdf(x)/sum + a3*rv3.pdf(x)/sum + a4*rv4.pdf(x)/sum )*(degree/2 + 1))

    target_graph = axes.get_graph(
            lambda x: 2*func(x),
            color=BLUE,
        )
    
    p = ValueTracker(0)

    v_approx = axes.get_graph(
            lambda x: 2*func2(x,  1) ,
            color = RED_E #,
            #x_min = 0,
            #x_max = 3
        )
    
    v_approx2 = axes.get_graph(
            lambda x: 2*func2(x,  1) ,
            color = RED_E #,
            #x_min = 0,
            #x_max = 3
        )

    v_approx.add_updater(
            lambda m: m.become(
                axes.get_graph(
                    lambda x: 2*func2(x,  p.get_value()),
                    color = RED_E 
                )
            )
        )

    text, number = label = VGroup(
            Tex(r"$\lambda = $").set_color(BLACK),
            DecimalNumber(
                0,
                show_ellipsis=True,
                num_decimal_places=1,
                include_sign=False,
            ).set_color(BLACK)
        )
    label.arrange(RIGHT)
    f_always(number.set_value, p.get_value)
    always(label.next_to, axes, DOWN)

    #target_label = axes.get_graph_label(target_graph, Tex(r"p(\\boldsymbol{\\vartheta}| \\boldsymbol{x}, \\boldsymbol{y})")).move_to(0.1*LEFT + UP).scale(0.5)
    target_label = Tex(r"""\fontfamily{lmss}\selectfont target density $p(\boldsymbol{\vartheta}| \boldsymbol{x}, \boldsymbol{y})$""").set_color(BLUE).scale(0.5)
    v_approx_label = Tex(r""" \fontfamily{lmss}\selectfont variational density $q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})$""").set_color(RED_E).scale(0.5)
    always(v_approx_label.next_to, target_graph, .5*RIGHT)
    always(target_label.next_to, target_graph, UP)
    #v_approx_label = axes.get_graph_label(v_approx, r"q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})").move_to(2*RIGHT + UP).scale(0.5)

    citation = Tex(r"Ong, V. M. H., Nott, D. J. and Smith, M. S. (2018). Gaussian variational approximation with a factor co-variance structure. \textit{Journal of Computational and Graphical Statistics}").set_color(BLACK).scale(0.3).move_to(2.75*DOWN + 3.5*LEFT).align_to(LEFT)
    final_graph = VGroup(axes, target_graph,  v_approx2) #target_label, v_approx_label,
    self.add(axes, title) 
    self.play(
            Create(target_graph),
            Create(target_label),
        )
    self.slide_break()
    #self.wait(2)
    self.play(Create(v_approx),
              Create(v_approx_label),
              Create(label))
    self.slide_break()
    
    self.play(p.animate.set_value(4), run_time=3)
    self.slide_break()
    self.remove(axes, target_graph, v_approx, label)
    self.play(final_graph.animate.move_to(3.5*LEFT))#, title.animate.move_to(4*LEFT + 2.5*UP)
    self.add(citation)
    self.slide_break()
    self.add(member)
    self.slide_break() 
    self.add(KLD)
    self.slide_break() 
    self.play(ReplacementTransform(KLD, vlb))
    self.slide_break() 
    self.play(Create(delta_vlb))
    self.slide_break() 
    self.play(Create(delta_vlb_update))
    self.wait(0.5)

class Data(SlideScene):
    def construct(self):
        set_background(self, "Data", True)
        title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Data \& End-to-End Learner}").move_to( 2.75*UP).set_color(BLACK).scale(0.7)


        ex = ImageMobject('files/image_road.png').scale(0.3).move_to(3*RIGHT + 1.25*UP)
        
        data0 = Tex(r"\fontfamily{lmss}\selectfont \textbf{comma2k19 data (Schafer et. al, 2018)}:").set_color(BLACK).scale(0.6).move_to(2*UP + 2.5*LEFT)
        data1 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize} \item 33 hours of video data on US highways with steering angles \end{itemize}""").set_color(BLACK).scale(0.5).move_to(1.75*LEFT + 1.5*UP)
        data2 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item extracted $n = 450,000$ frames \end{itemize}").set_color(BLACK).scale(0.5).move_to(1*UP).align_to(data1, LEFT)
        data3 = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item clean and raw version \end{itemize} ").set_color(BLACK).scale(0.5).move_to(.5*UP).align_to(data1, LEFT)
        pilot = ImageMobject('files/PilotNet.png').scale(.8).move_to(1.75*DOWN)
        pilot_text = Tex(r"\fontfamily{lmss}\selectfont \textbf{PilotNet Model Architecture} (Bojarski et. al, 2016)").move_to(pilot.get_top() + .2*UP + 2.5*LEFT).set_color(BLACK).scale(0.6).align_to(data0, LEFT)

        citations = VGroup(Tex(r"Bojarski,  M. et. al. (2016). End to end learning for self-driving cars."),
        Tex(r"Schafer, H., Santana, E., Haden, A. and Biasini, R. (2018). A commute in data: the comma2k19 dataset.")).arrange(DOWN).move_to(3*DOWN + 0*RIGHT).scale(0.3).set_color(BLACK).align_to(LEFT)
        self.add(ex, title, data0, data1, data2, data3)
        self.add(pilot, pilot_text)
        self.add(citations)
        self.wait(0.5)

class FullAlgorithm(SlideScene):
    def construct(self):
        set_background(self, "Data", True)
        title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Full Algorithm}").move_to( 2.75*UP).set_color(BLACK).scale(0.7)
        bullets = Tex(r"""\fontfamily{lmss}\selectfont
        \begin{enumerate}
        \item Estimate margin $F_Y$ from all steering angles in data $\boldsymbol{y}= (y_1,...,y_n)^T$
        \item Train PilotNet DNN to predict the transformed steering angles $z_i = \Phi^{-1}(F_Y(y_i))$ from images of the ahead-lying street $\boldsymbol{x}_i$, $i = 1,...,n$
        \item Obtain posterior of model parameters $p(\boldsymbol{\beta}, \boldsymbol{\theta}|  \boldsymbol{x}, \boldsymbol{y})$ via MCMC and VI using deep basis functions from previous step
        \item Compute predictive densities for new covariates $\boldsymbol{x}_0$ at value $y_0$ via $p(y_0 | \boldsymbol{x}_0, \boldsymbol{x}, \boldsymbol{y}) = \int p(y_0 | \boldsymbol{x}_0, \boldsymbol{\beta}, \boldsymbol{\theta}, \boldsymbol{x}, \boldsymbol{y})p(\boldsymbol{\beta}, \boldsymbol{\theta}| \boldsymbol{x}, \boldsymbol{y})d(\boldsymbol{\beta}, \boldsymbol{\theta})$
        \end{enumerate}""").set_color(BLACK).scale(0.5)
        
        bullets2 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize}
         \item[$\rightarrow$] now check accuracy of VI/MCMC \& quality of predictive densities
         \end{itemize} """).set_color(BLUE_D).scale(0.7).move_to(bullets.get_bottom() + 1*DOWN)
        self.add(title, bullets, bullets2)
        self.wait(0.5)

class PostMeanSD(SlideScene):
    def construct(self):
        set_background(self, "Accuracy VI vs. HMC (Horseshoe Prior)", True)

        hs_means = ImageMobject('files/accuracy_horseshoe/hs_means.png')
        hs_means.width = 6
        hs_means.height = 4

        hs_sd = ImageMobject('files/accuracy_horseshoe/hs_sd.png')
        hs_sd.width = 6
        hs_sd.height = 4

        hs_plots = Group(hs_means,hs_sd).arrange() #.move_to(np.array([5, -3.5, 0]))
        
        title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Accuracy VI vs. HMC (Horseshoe Prior)}").move_to( 2.75*UP).set_color(BLACK).scale(0.7)
        
        conclusion = Text("Overall VI is sufficiently accurate to substitute MCMC", font="Noto Sans").set_color(BLACK).move_to(hs_plots.get_bottom() + .75*DOWN).scale(0.5)
        conclusion.bg = SurroundingRectangle(conclusion, color=GREEN, fill_color=GREEN_A, fill_opacity=.2)

        self.add(title, hs_plots)
        self.slide_break()
        self.play(Create(conclusion), Create(conclusion.bg))
        self.wait(0.5)

class Calibration(SlideScene):
    def construct(self):
        set_background(self, "Calibration", True)

        marg_cal = ImageMobject('files/calibration/marginal_calibration.png')
        marg_cal.width = 6
        marg_cal.height = 4

        cov_rates = ImageMobject('files/prediction_intervals/coverage_rates.png')
        cov_rates.width = 6
        cov_rates.height = 4

        cal_plots = Group(marg_cal,cov_rates).arrange() 
        
        title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Reliability of the Predictive Densities}""").move_to(2.75*UP).set_color(BLACK).scale(0.7)
        
        conclusion = Text("The IC-NLM is well-calibrated and provides accurate coverage rates", font="Noto Sans").set_color(BLACK).move_to(cal_plots.get_bottom() + .75*DOWN).scale(0.5)
        conclusion.bg = SurroundingRectangle(conclusion, color=GREEN, fill_color=GREEN_A, fill_opacity=.2)

        self.add(title, cal_plots)
        self.slide_break()
        self.play(Create(conclusion), Create(conclusion.bg))
        self.wait(0.5)

class BenchmarkStudy(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)

        title = Tex(r"""\fontfamily{lmss}\selectfont Benchmark Study""").set_color(BLACK).move_to(2.75*UP).set_color(BLACK).scale(0.7)

        bullets = Tex(r"""\fontfamily{lmss}\selectfont \begin{enumerate}
        \item IC-NLM $+$ ridge prior
        \item IC-NLM $+$ horseshoe prior
        \item MC-dropout
        \item Mixture density network
        \end{enumerate} """).set_color(BLACK).scale(0.7)

        self.add(title, bullets)
        self.wait(0.5)
    

class PredictionIntervals(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)

        cov_rates = ImageMobject('files/prediction_intervals/coverage_rates.png')
        cov_rates.width = 6
        cov_rates.height = 4

        errconf = ImageMobject('files/prediction_intervals/error_vs_confidence.png')
        errconf.width = 6
        errconf.height = 4

        cal_plots = Group(cov_rates, errconf).arrange() #.move_to(np.array([5, -3.5, 0]))
        
        title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Prediction Intervals}").move_to(2.75*UP).set_color(BLACK).scale(0.7)
        
        conclusion = Text("The IC-NLM provides accurate coverage rates but improvable early warning properties ", font="Noto Sans").set_color(BLACK).move_to(cal_plots.get_bottom() + .75*DOWN).scale(0.5)
        conclusion.bg = SurroundingRectangle(conclusion, color=RED, fill_color=RED_A, fill_opacity=.2)

        self.add(title, cal_plots)
        self.slide_break()
        self.play(Create(conclusion), Create(conclusion.bg))
        self.wait(0.5)

class OutlookDiscussion(SlideScene):
    def construct(self):
        set_background(self, " Discussion & Outlook", True)

        title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Discussion \& Outlook}").move_to(2.75*UP ).set_color(BLACK).scale(0.7)
        bullet_points00 = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Implicit copula NLM for EtE learning}""").scale(.7).move_to(1.5*UP +1.5*LEFT).set_color(BLACK)
        bullet_points0 = Tex(r"""\fontfamily{lmss}\selectfont
         \begin{itemize}
         \item can be made scalable via variational inference
         \item is marginally calibrated and competitively prob. calibrated
         \item provides informative and reliable prediction intervals
         \end{itemize}
         """).move_to(.5*UP + 1.5*LEFT).set_color(BLACK).scale(0.5)
        
        bullet_points000 = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Outlook}""").move_to(1*DOWN).set_color(BLACK).scale(.7).align_to(bullet_points0, LEFT)
        bullet_points = Tex(r"""\fontfamily{lmss}\selectfont
         \begin{itemize} 
        \item Identify several steering actions to combine densities with route planning
        \item Integrate estimation into loss function
        \item Loss of information of neural linear models vs. full model uncertainty
        \end{itemize}
        """).set_color(BLACK).scale(0.5).move_to(2*DOWN + 1.5*LEFT).align_to(bullet_points0, LEFT)
        self.add(title, bullet_points, bullet_points0, bullet_points00, bullet_points000)
        self.wait(0.5)