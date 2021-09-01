from manim_slide import *
import imageio
import math
from scipy.stats import norm


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
      line = Line(start = np.array([5, 3, 0]), 
                  end = np.array([-4, 3, 0]), stroke_width=0.5).set_color(GRAY)
      section_title = Text(section, font="Noto Sans").scale(0.3).move_to(np.array([-5.5, 3, 0])).set_color(GRAY)
      self.add(line, section_title)
    self.add(logos)

class Title(SlideScene):
    def construct(self):
        set_background(self, None, True)
        title1 = Tex(r"\fontfamily{lmss}\selectfont \textbf{Marginally Calibrated Predictive Distributions} ").move_to(1*UP).scale(1.2).set_color(BLUE_E)
        title2 = Tex(r"\fontfamily{lmss}\selectfont \textbf{for End-to-End Learners in Autonomous Driving}").move_to(.4*UP).scale(1.2).set_color(BLUE_E)
        #title3 = Tex("in Autonomous Driving", font="Noto Sans").move_to(.4*UP).scale(0.8).set_color(BLUE_E)

        authors = Tex(r"\fontfamily{lmss}\selectfont Clara Hoffmann and Nadja Klein").scale(0.8).set_color(BLACK).move_to(DOWN)
        group = Tex(r"\fontfamily{lmss}\selectfont Emmy Noether Research Group in Statistics \& Data Science, Humboldt-Universität zu Berlin").scale(0.5).set_color(BLACK).move_to(1.5*DOWN)

        date = Tex(r"\fontfamily{lmss}\selectfont September 16th, 2021").set_color(BLACK).scale(0.6).move_to(2.5*DOWN)
        conference = Tex(r"\fontfamily{lmss}\selectfont Statistische Woche 2021").set_color(BLACK).scale(0.6).move_to(2.75*DOWN)
        self.add(title1, title2,  authors, group, date, conference) # title3,
        self.wait(0.5)

class EtELearning(SlideScene):
    def construct(self):
        set_background(self, "End-to-End Learning", True)
        frame_title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{End-to-End Learning} """).move_to(2*UP).set_color(BLACK).scale(0.7)
        ete_diag = ImageMobject('files/ete_diagram/ete.png').move_to(.25*DOWN)
        ete_diag.width = 6
        ete_diag.height = 4
        self.add(frame_title, ete_diag)
        self.wait(0.5)

class Motivation(SlideScene):
    def construct(self):
        set_background(self, "Motivation", True)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage[T1]{fontenc}")
        myTemplate.add_to_preamble(r"\usepackage{lmodern}")
        
        
        title = Tex(r"\fontfamily{lmss}\selectfont \begin{itemize} \item  Shortcomings of current methods for uncertainty quantification: \end{itemize}").move_to(np.array([-2, 1, 0])).set_color(BLACK).scale(0.5)
        problems1 = Tex(r""" \fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item not scalable \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).move_to(.5*UP + 3*LEFT)
        problems2 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item have to be evaluated in parallel  \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).move_to(0*UP +  2*LEFT)
        problems3 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item strong parametric assumptions \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).move_to(-.5*UP + 2*LEFT)
        problems4 = Tex(r"""\fontfamily{lmss}\selectfont \begin{itemize} \begin{itemize} \item not calibrated \end{itemize} \end{itemize}}""").scale(0.5).set_color(BLACK).move_to(-1*UP + 2.9*LEFT)

        
        solution = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Solution}: Implicit-copula neural linear model of Klein, Nott, Smith 2020 """).set_color(BLACK).scale(.7).align_to(title, LEFT).move_to(2*DOWN)
        solution.bg = SurroundingRectangle(solution, color=GREEN_D, fill_color=GREEN_A, fill_opacity=.2)

        frame_title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Problems in Uncertainty Quantification for EtE Learning} """).move_to(2*UP).set_color(BLACK).scale(0.7)

        self.add(frame_title)
        self.slide_break()
        self.add(title, problems1)
        self.slide_break()
        self.add(problems2)
        self.slide_break()
        self.add(problems3)
        self.slide_break()
        self.add(problems4)
        self.slide_break()
        self.add(solution, solution.bg)
        self.wait(0.5)

class Notation(SlideScene):
    def construct(self):
        set_background(self, "Variables and Notation", True)
        placeholder = Tex("Placeholder notation").set_color(BLACK)
        self.add(placeholder)
        self.wait(0.5)

class NLM(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)
        placeholder = Tex("Placeholder neural linear model").set_color(BLACK)
        self.add(placeholder)
        self.wait(0.5)

class CopulaSlide(SlideScene):
  def construct(self):
    set_background(self, "Copula Model", True)


    title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Implicit Copula Neural Linear Model (IC-NLM)} """).move_to(2*UP).set_color(BLACK).scale(0.9)

    nlm_text = Tex(r"\fontfamily{lmss}\selectfont pseudo regression: ").set_color(BLACK).scale(0.9)
    nlm1 = Tex(r"$\Tilde{Z} = $").set_color(BLACK) #.move_to(RIGHT)
    nlm2 = Tex(r"$B_{\boldsymbol{\zeta}}(\boldsymbol{x})\boldsymbol{\beta}$").set_color(BLACK).scale(0.9)
    nlm3 = Tex(r"$ + \boldsymbol{\varepsilon}$").set_color(BLACK).scale(0.79)
    nlm = VGroup(VGroup(nlm_text, nlm1).arrange(RIGHT, buff=MED_SMALL_BUFF), nlm2, nlm3).arrange(RIGHT, buff=SMALL_BUFF).move_to(np.array([-4, 1, 0])).scale(0.7)

    nlm_text_dbf_text = Tex(r"\fontfamily{lmss}\selectfont deep basis functions").move_to(nlm2.get_bottom() + DOWN).set_color(BLACK).scale(0.6)
    nlm_text_dbf_line =  Line(nlm2.get_bottom(), nlm_text_dbf_text.get_top(), stroke_width = 1).set_color(BLACK)
    nlm_text_dbf = VGroup(nlm_text_dbf_text, nlm_text_dbf_line)
    
    nlm_text_error_text = Tex(r"\fontfamily{lmss}\selectfont artificial error term").move_to(nlm3.get_bottom()+ 2*DOWN + 0.5*RIGHT).set_color(BLACK).scale(0.6)
    nlm_text_error_line =  Line(nlm3.get_bottom() + .5*RIGHT,nlm_text_error_text.get_top(), stroke_width = 0.5).set_color(BLACK)
    nlm_text_error = VGroup(nlm_text_error_text, nlm_text_error_line)

    #nlm2 = nlm.copy().move_to(np.array([-1.5, 2, 0])).scale(0.7)


    errors = Tex(r"$\varepsilon_i \sim N(0, \sigma^2)$").set_color(BLACK)
    errors2 = errors.copy().move_to(nlm2.get_right()+ 1.5*RIGHT).scale(0.7)
    errors2_text_text = Text(r"\fontfamily{lmss}\selectfont i.i.d.").move_to(errors.get_bottom()+ DOWN).set_color(BLACK).scale(0.6)
    errors2_line =  Line(errors.get_bottom(),errors2_text_text.get_top(), stroke_width = 0.5).set_color(BLACK)
    errors2_text = VGroup(errors2_text_text, errors2_line)


    beta_text = Text(r"\fontfamily{lmss}\selectfont shrinkage prior: ").move_to(RIGHT).scale(0.7)
    beta1 = Tex(r" \quad $\boldsymbol{\beta} | \boldsymbol{\theta}, \sigma^2 \sim N(0, \sigma^2$").set_color(BLACK)
    beta12 = Tex(r"$P(\boldsymbol{\theta})^{-1})$").set_color(BLACK)
    beta = VGroup(VGroup(beta_text, beta1).arrange(RIGHT, buff=MED_SMALL_BUFF), beta12).arrange(RIGHT, buff=SMALL_BUFF)
    beta2 = beta.copy().move_to(np.array([0, 1, 0])).scale(0.7)
    beta1_text_text = Text(r"\fontfamily{lmss}\selectfont shrinkage parameters").move_to(beta12.get_bottom() + DOWN).set_color(BLACK).scale(0.6)
    beta1_line =  Line(beta12.get_bottom(), beta1_text_text.get_top(), stroke_width = 1).set_color(BLACK)
    beta_text = VGroup(beta1_text_text, beta1_line)

    #z_post = Tex(r"$\Tilde{Z} | \boldsymbol{x}, \sigma^2, \boldsymbol{\theta} \sim N(0, \sigma^2 \Sigma$").set_color(BLACK)
    z = Tex(r"$ Z = \sigma^{-1}S(\boldsymbol{x}, \boldsymbol{\theta}) \Tilde{Z} | \boldsymbol{x}, \sigma^2 \sim N(0, R(\boldsymbol{x}, \boldsymbol{\theta}))$").set_color(BLACK).move_to(np.array([-1, -1, 0]))
    #all_z = VGroup(z_post, z).arrange(RIGHT)
    z_text_text = VGroup(Tex(r"$\boldsymbol{\beta}$"), Text(r"\fontfamily{lmss}\selectfont integrated out").scale(0.7)).arrange(RIGHT).move_to(z.get_bottom() + DOWN).set_color(BLACK).scale(0.8)
    z_text_line =  Line(z.get_bottom(), z_text_text.get_top(), stroke_width = 1).set_color(BLACK)
    z_text = VGroup(z_text_text, z_text_line)

    z_text_text2 = VGroup(Tex(r"\fontfamily{lmss}\selectfont margins").scale(0.7), Tex(r"$\sim N(0,1)$")).arrange(RIGHT).move_to(z.get_bottom() + DOWN + 4*RIGHT).set_color(BLACK).scale(0.8)
    z_text_line2 =  Line(z.get_bottom() + 4*RIGHT, z_text_text2.get_top(), stroke_width = 1).set_color(BLACK)
    z_text2 = VGroup(z_text_text2, z_text_line2)
    all_z2 = z.copy().move_to(np.array([0, 0, 0])).scale(0.7)


    rectangle = Rectangle(height=2, width=3)

    #sklars = Text(r"Sklar's theorem", font="Noto Sans").set_color(BLACK).scale(0.5).move_to(.5*UP)
    sklars = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Sklar's theorem} """).move_to(2*UP).set_color(BLACK).scale(0.7)
    n = 2
    sklar_eq = Tex(r"$p(\boldsymbol{z} | \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$ = $",
                   r"$c_{DNN}(F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})$",
                   r"$| \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$\prod_{i=1}^n$",
                   r"$p_{z_i}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$"
                      ).move_to(sklars.get_bottom() + n*DOWN).set_color(BLACK).scale(0.8)
    
    sklar_eq_col = sklar_eq.copy()
    sklar_eq_col.set_color_by_tex_to_color_map({
        r"$p(\boldsymbol{z} | \boldsymbol{x}, \boldsymbol{\theta})$": RED_E,
        r"$p_{z_i}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$": RED_E
             }) 

    sklar_eq2 = Tex(r"$\phi_n(\boldsymbol{z}; \boldsymbol{0}, R(\boldsymbol{x}, \boldsymbol{\theta}))$",
                   r"$ = $",
                   r"$c_{DNN}(F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})$",
                   r"$| \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$\prod_{i=1}^n $",
                   r"$\phi_{1}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$"
                      ).move_to(sklars.get_bottom() + n*DOWN).set_color(BLACK).scale(0.8) 
    sklar_eq2.set_color_by_tex_to_color_map({
        r"$\phi_n(\boldsymbol{z}; \boldsymbol{0}, R(\boldsymbol{x}, \boldsymbol{\theta})$": RED_E,
        r"$\phi_{1}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})$": RED_E
             })   
    
    # should be nicer to move...
    copula_dens = Tex(r"$c_{DNN}(F_{z_1}(z_1 | \boldsymbol{x}_1, \boldsymbol{\theta}), \ldots,  F_{z_n}(z_n | \boldsymbol{x}_n, \boldsymbol{\theta})| \boldsymbol{x}, \boldsymbol{\theta})$",
                   r"$ = $",
                   r"$\frac{\phi_n(\boldsymbol{z}; \boldsymbol{0}, R(\boldsymbol{x}, \boldsymbol{\theta})}{\prod_{i=1}^n \phi_{1}(z_i| \boldsymbol{x}_i, \boldsymbol{\theta})}$"
                      ).move_to(sklars.get_bottom() + n*DOWN).set_color(BLACK).scale(0.8)    
    
    # Now we use the same copula to model y with the shrinkage parameters integrated out
    # note that even though the copula from before is a Gaussian copula
    # the copula with the shrinkage parameters integrated out can be far from Gaussian
    copula_dens_y = Tex(r"$p(\boldsymbol{y}| \boldsymbol{x}) = c_{DNN}^* ($",
                        r"$F_Y(y_1), \ldots, F_Y(y_n)$",
                         r"$| \boldsymbol{x}) \prod_{i=1}$",
                         r"$F_Y(y_i)$"
                   ).move_to(copula_dens.get_bottom() + n/2*DOWN).set_color(BLACK).scale(0.8)
    
    copula_dens_y.set_color_by_tex_to_color_map({
        r"$F_Y(y_1), \ldots, F_Y(y_n)$": RED_E,
        r"$F_Y(y_i)$": RED_E
             })
    
    # this copula density involves inverting an nxn matrix 
    # as a solution we evaluate the density of y conditional on the model parameters
    # and integrate over them, so that the final expression we're interest in is
    final_exp_title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Target Expression} """).move_to(2*UP).set_color(BLACK).scale(0.7)
    n = 2
    final_expr = Tex(r"$p(\boldsymbol{y} | \boldsymbol{x}) = \int p(\boldsymbol{y}| \boldsymbol{x}, \boldsymbol{\beta}, \boldsymbol{\theta}) $",
                     r"$p(\boldsymbol{\beta}, \boldsymbol{\theta}| \boldsymbol{x}, \boldsymbol{y})$",
                     r"$d(\boldsymbol{\beta}, \boldsymbol{y})$").set_color(BLACK).scale(0.8)
    
    final_expr.set_color_by_tex_to_color_map({
        r"$p(\boldsymbol{\beta}, \boldsymbol{\theta}| \boldsymbol{x}, \boldsymbol{y})$": RED_E
             })
    final_exp_text_text = Tex(r"known up to normalizing constant", font="Noto Sans").move_to(final_expr.get_bottom() + DOWN ).set_color(BLACK).scale(0.8)
    final_exp_text_line =  Line(final_expr.get_bottom(), final_exp_text_text.get_top(), stroke_width = 1).set_color(BLACK)
    final_exp_text = VGroup(final_exp_text_text, final_exp_text_line)
    
    # the posterior of the model parameters is only known up to a normalizing constant so it has to be approximated.
    # in former work, the posterior was estimated in MCMC
    # However, this is not possible in EtE learning, where we have immense data sets
    # With the data set sizes typical in EtE learning, MCMC samplers become slow or sometimes even don't converge at all

    # To ensure fast and scalable estimation we employ a variational inference approach

    # REGRESSION MODEL
    
    self.add(title, nlm)
    self.slide_break() #self.wait()
    self.add(nlm_text_dbf)
    self.slide_break() #self.wait()
    self.add(nlm_text_error)
    self.slide_break() #self.wait()
    self.remove(nlm_text_dbf, nlm_text_error)
    self.play(ReplacementTransform(nlm, nlm2))
    self.slide_break() #self.wait()

    # ERROR TERM
    self.add(errors)
    self.slide_break() #self.wait()
    self.add(errors2_text)
    self.slide_break() #self.wait()
    self.remove(errors2_text)
    self.slide_break() #self.wait()
    self.play(ReplacementTransform(errors, errors2))
    self.slide_break()

    # BETA
    self.add(beta)
    self.slide_break() #self.wait()
    self.add(beta_text)
    self.slide_break() #self.wait()
    self.remove(beta_text)
    self.play(ReplacementTransform(beta, beta2))
    self.slide_break()

    # beta integrated out
    self.add(z)
    self.slide_break() #self.wait()
    self.add(z_text)
    self.slide_break() #self.wait()
    self.add(z_text2)
    self.slide_break() #self.wait()
    self.remove(z_text, z_text2)
    self.play(ReplacementTransform(z, all_z2))
    self.slide_break() #self.wait()

    self.remove(nlm2, errors2, beta2, all_z2, z_text2, title)
    self.slide_break() #self.wait()

    # SKLARS THEOREM
    self.add(sklars, sklar_eq)
    self.slide_break() #self.wait()
    
    self.play(ReplacementTransform(sklar_eq, sklar_eq_col))
    self.slide_break() #self.wait(2)
    self.play(ReplacementTransform(sklar_eq_col, sklar_eq2))
    self.slide_break() #self.wait(2)
    self.play(ReplacementTransform(sklar_eq2, copula_dens))
    self.slide_break() #self.wait(2)
    self.add(copula_dens_y)
    self.slide_break() #self.wait(2)

    self.remove(copula_dens, copula_dens_y, sklars)
    self.play(ReplacementTransform(copula_dens_y, final_expr))
    self.add(final_exp_title)
    self.slide_break() #self.wait()
    self.add(final_exp_text)
    self.slide_break() #self.wait(2)


    #self.wait()
    #self.play(ReplacementTransform(nlm, nlm2))
    #self.wait()

    #self.add(errors)
    #self.wait()
    #self.play(ReplacementTransform(errors, errors2))
    #self.wait()

    #self.add(beta)
    #self.wait()
    #self.play(ReplacementTransform(beta, beta2))
    #self.wait()



class VIvsHMC(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)
        placeholder = Tex("Placeholder VI vs. HMC").set_color(BLACK)
        self.add(placeholder)
        self.wait(0.5)
        
class ExampleSlide(SlideScene):
    def construct(self):
        circle = Circle(radius=1, color=BLUE)
        dot = Dot()
        dot2 = dot.copy().shift(RIGHT)
        self.add(dot)

        line = Line([3, 0, 0], [5, 0, 0])
        self.add(line)

        self.play(GrowFromCenter(circle))
        self.slide_break()
        self.play(Transform(dot, dot2))
        self.slide_break()
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.slide_break()
        self.play(Rotating(dot, about_point=[2, 0, 0]), run_time=1.5)
        self.wait()


class VISlide(SlideScene):
  def construct(self):
    set_background(self, "Variational Inference", True)
    var_density = Tex(r"""$q_{\boldsymbol{\lambda}}(\boldsymbol{\beta}, \boldsymbol{\theta})$""").set_color(RED_E)
    member = VGroup(Text("approximation family:", font="Noto Sans").set_color(BLACK).scale(0.5),
                    Tex(r"""$q_{\boldsymbol{\lambda}}(\boldsymbol{\beta}, \boldsymbol{\theta}) = 
    N(\boldsymbol{\mu}, \boldsymbol{\Upsilon} \boldsymbol{\Upsilon}^T + \Delta^2)$""").set_color(BLACK)).arrange(DOWN).move_to(2*RIGHT).scale(0.8)


    KLD = VGroup(Text("Kullback-Leibler divergence:", font="Noto Sans").set_color(BLACK).scale(0.5),
                 Tex(r"""$\mathrm{KLD} (q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta}) || p(\boldsymbol{\vartheta}| \boldsymbol{y})) = 
    \int \frac{q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})}{p(\boldsymbol{\vartheta} | \boldsymbol{y})}
    q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})d(\boldsymbol{\vartheta})$""").set_color(BLACK)).arrange(DOWN).move_to(2*RIGHT).scale(0.8)

    vlb = VGroup(Text("variational lower bound:", font="Noto Sans").set_color(BLACK).scale(0.5),
                 Tex(r"""$\mathcal{L}(\boldsymbol{\lambda}) = 
    \mathbb{E}_{q_{\boldsymbol{\lambda}}} [\log(p(\boldsymbol{y}|\boldsymbol{\vartheta})p(\boldsymbol{\vartheta})) - 
    \log q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})] $""").set_color(BLACK)).arrange(DOWN).move_to(1*UP + 2*RIGHT ).scale(0.8)
    
    delta_vlb = VGroup(Text("optimize via gradient:", font="Noto Sans").set_color(BLACK).scale(0.5),
                       Tex(r"$\nabla_{\boldsymbol{\lambda}}\mathcal{L}(\boldsymbol{\lambda})$").set_color(BLACK)).arrange(DOWN).move_to(2*RIGHT).scale(0.8)
    
    delta_vlb_update =  VGroup(Text("update rule:", font="Noto Sans").set_color(BLACK).scale(0.5),
                               Tex(r"""$ \boldsymbol{\lambda}^{(t+1)} = 
    \boldsymbol{\lambda}^{(t)} + \rho 
    \nabla_{\boldsymbol{\lambda}}\mathcal{L}(\boldsymbol{\lambda}^{(t)})$""").set_color(BLACK)).arrange(DOWN).move_to(2*RIGHT).scale(0.8)

    #self.add(var_density)


    axes = Axes((0, 3), (0, 5)).scale(0.4).move_to(RIGHT)


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

    #area1 =  axes.get_area(target_graph, x_range=[0,3], color=BLUE, opacity = 0.2)
    #area2 =  axes.get_area(v_approx, x_range=[0,3], color=RED, opacity = 0.2)

    #area2.add_updater(lambda m: m.become(
    #    axes.get_area(v_approx, x_range=[0,3], color=RED, opacity = 0.2)))

    #target_label = axes.get_graph_label(target_graph, "p(\\boldsymbol{\\vartheta}| \\boldsymbol{x}, \\boldsymbol{y})").move_to(0.1*LEFT + UP).scale(0.5)
    #v_approx_label = axes.get_graph_label(v_approx, r"q_{\boldsymbol{\lambda}}(\boldsymbol{\vartheta})").move_to(2*RIGHT + UP).scale(0.5)

    final_graph = VGroup(axes, target_graph,  v_approx2) #target_label, v_approx_label,
    self.add(axes) #, area1
    self.play(
            Create(target_graph),
            #Create(area2),
            #Create(target_label),
        )
    self.slide_break()
    #self.wait(2)
    self.play(Create(v_approx),
             #Create(v_approx_label),
              Create(label))
    self.slide_break()
    
    self.play(p.animate.set_value(4), run_time=3)
    self.slide_break()
    #self.wait(2)
    self.remove(axes, target_graph, v_approx, label)
    self.play(final_graph.animate.move_to(4*LEFT))
    self.slide_break()
    #self.wait(2)
    self.add(member)
    self.slide_break() #self.wait()
    self.play(member.animate.move_to(2.5*UP + 2*RIGHT ))
    self.slide_break() # self.wait()
    self.add(KLD)
    self.slide_break() #self.wait()
    self.play(KLD.animate.move_to(1*UP + 2*RIGHT ))
    self.slide_break() #self.wait()
    self.play(ReplacementTransform(KLD, vlb))
    self.slide_break() #self.wait()
    self.add(delta_vlb.move_to(.5*DOWN + 2*RIGHT))
    self.slide_break() #self.wait()
    self.add(delta_vlb_update.move_to(2*DOWN + 2*RIGHT))
    self.wait()

class Data(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)
        placeholder = Tex("Placeholder neural linear model").set_color(BLACK)
        self.add(placeholder)
        self.wait(0.5)

class PostMeanSD(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage[T1]{fontenc}")
        myTemplate.add_to_preamble(r"\usepackage{lmodern}")

        hs_means = ImageMobject('files/accuracy_horseshoe/hs_means.png')
        hs_means.width = 6
        hs_means.height = 4

        hs_sd = ImageMobject('files/accuracy_horseshoe/hs_sd.png')
        hs_sd.width = 6
        hs_sd.height = 4

        hs_plots = Group(hs_means,hs_sd).arrange() #.move_to(np.array([5, -3.5, 0]))
        
        title = Tex(r"\fontfamily{lmss}\selectfontAccuracy VI vs. HMC").move_to(hs_plots.get_top() + .5*UP).set_color(BLACK).scale(0.5)
        
        conclusion = Text("The IC-NLM is better calibrated than MC-Dropout and the Mixture Density Network", font="Noto Sans").set_color(BLACK).move_to(hs_plots.get_bottom() + .75*DOWN).scale(0.5)
        conclusion.bg = SurroundingRectangle(conclusion, color=RED, fill_color=RED_A, fill_opacity=.2)

        self.add(title, hs_plots)
        self.slide_break()
        self.play(Create(conclusion), Create(conclusion.bg))
        self.wait(0.5)

class Calibration(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage[T1]{fontenc}")
        myTemplate.add_to_preamble(r"\usepackage{lmodern}")

        marg_cal = ImageMobject('files/calibration/marginal_calibration.png')
        marg_cal.width = 6
        marg_cal.height = 4

        prob_cal = ImageMobject('files/calibration/prob_calibration.png')
        prob_cal.width = 6
        prob_cal.height = 4

        cal_plots = Group(marg_cal,prob_cal).arrange() 
        
        title = Tex(r"""\fontfamily{lmss}\selectfont \textbf{Marginal \& Probability Calibration}""").move_to(cal_plots.get_top() + .5*UP).set_color(BLACK).scale(0.5)
        
        conclusion = Text("The IC-NLM is better calibrated than MC-Dropout and the Mixture Density Network", font="Noto Sans").set_color(BLACK).move_to(cal_plots.get_bottom() + .75*DOWN).scale(0.5)
        conclusion.bg = SurroundingRectangle(conclusion, color=RED, fill_color=RED_A, fill_opacity=.2)


        #placeholder = Tex("Placeholder neural linear model").set_color(BLACK)
        self.add(title, cal_plots)
        self.slide_break()
        self.play(Create(conclusion), Create(conclusion.bg))
        self.wait(0.5)

class PredictionIntervals(SlideScene):
    def construct(self):
        set_background(self, "Neural Linear Model", True)
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage[T1]{fontenc}")
        myTemplate.add_to_preamble(r"\usepackage{lmodern}")

        cov_rates = ImageMobject('files/prediction_intervals/coverage_rates.png')
        cov_rates.width = 6
        cov_rates.height = 4

        errconf = ImageMobject('files/prediction_intervals/error_vs_confidence.png')
        errconf.width = 6
        errconf.height = 4

        cal_plots = Group(cov_rates, errconf).arrange() #.move_to(np.array([5, -3.5, 0]))
        
        title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Prediction Intervals}").move_to(cal_plots.get_top() + .5*UP).set_color(BLACK).scale(0.5)
        
        conclusion = Text("The IC-NLM provides accurate coverage rates but improvable early warning properties ", font="Noto Sans").set_color(BLACK).move_to(cal_plots.get_bottom() + .75*DOWN).scale(0.5)
        conclusion.bg = SurroundingRectangle(conclusion, color=RED, fill_color=RED_A, fill_opacity=.2)

        self.add(title, cal_plots)
        self.slide_break()
        self.play(Create(conclusion), Create(conclusion.bg))
        self.wait(0.5)

class OutlookDiscussion(SlideScene):
    def construct(self):
        set_background(self, "Outlook \& Discussion", True)
        title = Tex(r"\fontfamily{lmss}\selectfont \textbf{Outlook and Discussion}").move_to(2*UP).set_color(BLACK).scale(0.7)
        placeholder = Tex(r"""\begin{itemize}
        \item Regularized horseshoe prior
        \item Combine densities with route planning
        \item Identify several steering actions
        \item Integrate VI into training
        \item Loss of information of neural linear models vs. full model uncertainty
        \end{itemize}
        """).set_color(BLACK)
        self.add(placeholder)
        self.wait(0.5)