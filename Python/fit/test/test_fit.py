import unittest as ut
import numpy as np
from numpy.random import random_sample
from numpy import testing as nptest
from .. import fit

class TestFit(ut.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_validate_knots_and_data_too_few_knots(self):
        p = 2
        knots = np.arange(2)
        x = np.arange(0, 1, 11)
        self.assertRaisesRegex(ValueError, r'There must be at least 5 knots', fit.validate_knots_and_data, p=p,
                               knots=knots, x=x)

    def test_validate_knots_and_data_decreasing_knots(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .4, .6, 1, 1, 1]
        x = np.linspace(0,1, 11)
        self.assertRaises(ValueError, fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_decreasing_x(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = [0, .1, .2, .3, .5, .4, .6, .7, .8, .9, 1]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_data_low_x(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = [-1]
        self.assertRaisesRegex(ValueError, r'outside the knot', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_data_high_x(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = [2]
        self.assertRaisesRegex(ValueError, r'outside the knot', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_sites_no_data_in_nonzero_span(self):
        p = 2
        knots = [0, 0, 0, .3, .5, .5, .6, 1, 1, 1]
        x = np.linspace(.3, 1, 8)
        self.assertRaisesRegex(ValueError, r'no constraining data', fit.validate_knots_and_data, p=p, knots=knots, x=x)

    def test_validate_knots_and_sites_one_site_per_basis_function(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = [.1, .3, .6]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_sites_last_span_only_has_lastknot(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .6, 1, 1, 1])
        x = [.1, .5, 1]
        self.assertIsNone(fit.validate_knots_and_data(p, knots, x))

    def test_validate_knots_and_sites_excess_multiplicity(self):
        p = 2
        knots = np.array([0, 0, 0, .3, .5, .5, .5, .5, .6, 1, 1, 1])
        x = np.linspace(0,1,11)
        self.assertRaisesRegex(ValueError, r'multiplicity greater than 3', fit.validate_knots_and_data,
                               p=p, knots=knots, x=x)

    def test_get_default_interior_knots_p2(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        nptest.assert_array_equal([7.0, 9.0], fit.get_default_interior_knots(p,x))

    def test_get_default_interior_knots_p3(self):
        p = 3
        x = [5, 6, 8, 10, 40, 100]
        nptest.assert_array_almost_equal([8, 58/3], fit.get_default_interior_knots(p,x))

    def test_augment_knots_first_last_with_correct_multiplicity(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        iknots = [8, 12]
        nptest.assert_array_equal([5,5,5,8,12,40,40,40], fit.augment_knots(p, iknots, x))

    def test_get_default_knots_p2(self):
        p = 2
        x = [5, 6, 8, 10, 40]
        nptest.assert_array_equal([5,5,5,7,9,40,40,40], fit.get_default_knots(p,x))

    def test_get_default_knots_p3(self):
        p = 3
        x = [5, 6, 8, 10, 40]
        nptest.assert_array_almost_equal([5, 5, 5, 5, 8, 40, 40, 40, 40], fit.get_default_knots(p, x))

    def test_get_spline_quadratic_noisy_parabola_no_regularization(self):
        p = 2
        x = np.linspace(0, 10, 21)
        y = np.poly1d([3, 2, 1])(x) + 1e-2*random_sample(len(x))
        knots = [0, 0, 0, 2.5, 5, 7.5, 10, 10, 10]
        bsp = fit.get_spline(p, knots, x, y)
        nptest.assert_array_equal(knots, bsp._knots)
        nptest.assert_array_equal(p, bsp._degree)
        self.assertEqual(6, len(bsp._coefs))
        nptest.assert_array_almost_equal([1.003, 3.506, 46.006, 126.007, 243.506, 321.002], bsp._coefs, decimal=2)

    def test_get_spline_quadratic_ignores_regularization_above_p(self):
        p = 2
        x = np.linspace(0, 10, 21)
        y = np.poly1d([3, 2, 1])(x) + 1e-2 * random_sample(len(x))
        knots = [0, 0, 0, 2.5, 5, 7.5, 10, 10, 10]
        bsp = fit.get_spline(p, knots, x, y)
        bspd = fit.get_spline(p, knots, x, y, minimize_d3_x=[5,6])
        nptest.assert_array_equal(bsp._coefs, bspd._coefs)

    def test_get_spline_quadratic_with_d1_regularization(self):
        p = 2
        x = [.197, 1, 3, 7, 20, 27, 39]
        y = [.5, .59, .73, 1.1, 2.1, 2.2, 1.7]
        min_d1_x = [.197, 27]
        knots = fit.augment_knots(2, [.5, 1, 5, 10, 30], x)
        bsp = fit.get_spline(p, knots, x, y)
        bspr = fit.get_spline(p, knots, x, y, minimize_d1_x=min_d1_x)
        nptest.assert_array_equal(knots, bsp._knots)
        nptest.assert_array_equal(p, bsp._degree)
        self.assertEqual(8, bsp._coefs.size)
        self.assertEqual(8, bspr._coefs.size)
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, bsp._coefs, bspr._coefs)

    def test_get_spline_cubic_with_d1d2_regularization(self):
        p = 3
        x = [.197, 1, 3, 7, 20, 27, 39, 197]
        y = [.5, .59, .73, 1.1, 2.1, 2.2, 1.7, 1.4]
        min_d1_x = [.197, 27, 197]
        min_d2_x = np.linspace(40, 197)
        int_knots = [.5, 1, 5, 10, 30]
        knots1 = fit.augment_knots(2, int_knots, x)
        bsp = fit.get_spline(p, knots1, x, y)
        bsp1r = fit.get_spline(p, knots1, x, y, minimize_d1_x=min_d1_x)
        knots2 = fit.augment_knots(2, np.concatenate((int_knots, np.linspace(40, 195, 20))), x)
        bsp2r = fit.get_spline(p, knots2, x, y, minimize_d1_x=min_d1_x, minimize_d2_x=min_d2_x)
        nptest.assert_array_equal(knots2, bsp2r._knots)
        self.assertRaises(AssertionError, nptest.assert_array_almost_equal, bsp1r._coefs, bsp2r._coefs)
        # from matplotlib import pyplot as plt
        # plt.plot(x, y, '*',label='data')
        # pltx = np.linspace(x[0],x[-1],2000)
        # plt.plot(pltx,bsp(pltx),label='fit - no regularization')
        # plt.plot(pltx,bsp1r(pltx),label='fit with d1 regularization')
        # plt.plot(pltx,bsp2r(pltx),label='fit with d1 and d2 regularization')
        # plt.legend()
        # plt.show(block=True)

    def test_energy_fit(self):
        dat = np.array([(1.0e-7, 2.85E+10),
                        (1.0e-6, 2.85E+10),
                        (1.0e-5, 2.85E+10),
                        (1.0e-4, 2.85E+10),
                        (1.0e-3, 2.85E+10),
                        (1.0e-2, 2.85E+10),
                        (1.0e-1, 2.85E+10),
                        (3.274974164413447E+00, 8.888256622928249E+09),
                        (3.334372060618770E+00, 8.503306907046059E+09),
                        (3.394847250841235E+00, 8.119211909193420E+09),
                        (3.456419273860388E+00, 7.736357613483673E+09),
                        (3.519108022828766E+00, 7.355148715730030E+09),
                        (3.582933751699131E+00, 6.976009135334621E+09),
                        (3.647917081768267E+00, 6.599382565751526E+09),
                        (3.714079008339465E+00, 6.225733068602276E+09),
                        (3.781440907505846E+00, 5.855545716842644E+09),
                        (3.850024543056699E+00, 5.489327293089523E+09),
                        (3.919852073509108E+00, 5.127607050202980E+09),
                        (3.990946059267072E+00, 4.770937541660305E+09),
                        (4.063329469910512E+00, 4.419895530041206E+09),
                        (4.137025691616432E+00, 4.075082983435497E+09),
                        (4.212058534714722E+00, 3.737128170677120E+09),
                        (4.288452241380967E+00, 3.406686859886353E+09),
                        (4.366231493468793E+00, 3.084443505178776E+09),
                        (4.445421420484270E+00, 2.771111685497135E+09),
                        (4.526047607704936E+00, 2.467436876519325E+09),
                        (4.608136104446089E+00, 2.174616123935863E+09),
                        (4.691713432476985E+00, 1.893501149617615E+09),
                        (4.776806594589700E+00, 1.623948497649490E+09),
                        (4.863443083323393E+00, 1.366846193355408E+09),
                        (4.951650889846801E+00, 1.123088875177822E+09),
                        (5.041458513001834E+00, 8.936038777038460E+08),
                        (5.132894968511204E+00, 6.793561257736626E+08),
                        (5.225989798353033E+00, 4.813501903017578E+08),
                        (5.320773080305502E+00, 3.006326907713470E+08),
                        (5.417275437664607E+00, 1.383052919571915E+08),
                        (5.515528049138164E+00, -4.332213633079529E+06),
                        (5.615562658919256E+00, -1.242522067661934E+08),
                        (5.717411586942391E+00, -2.188633471504555E+08),
                        (5.821107739325670E+00, -2.878026647389908E+08),
                        (5.910000000000000E+00, -3.244100440756798E+08),
                        (5.998130873340072E+00, -3.404823089587097E+08),
                        (6.106918404830235E+00, -3.327487733383713E+08),
                        (6.217679005474378E+00, -2.934182901556435E+08),
                        (6.330448460640852E+00, -2.200169566763115E+08),
                        (6.445263204733846E+00, -1.113012981602974E+08),
                        (6.562160332964882E+00, 3.500380454063416E+07),
                        (6.681177613337825E+00, 2.210633187820969E+08),
                        (6.802353498851243E+00, 4.486868507965317E+08),
                        (6.925727139922103E+00, 7.183844449005775E+08),
                        (7.051338397034772E+00, 1.031433376489056E+09),
                        (7.179227853619443E+00, 1.393044084876152E+09),
                        (7.309436829164150E+00, 1.806229727619568E+09),
                        (7.442007392564554E+00, 2.273755555375454E+09),
                        (7.576982375715899E+00, 2.798590444345375E+09),
                        (7.714405387351456E+00, 3.383822515105461E+09),
                        (7.854320827131956E+00, 4.032620217765045E+09),
                        (7.996773899990580E+00, 4.748243600233395E+09),
                        (8.141810630738089E+00, 5.534046551855835E+09),
                        (8.289477878932882E+00, 6.393476682375992E+09),
                        (8.439823354020731E+00, 7.330075201515793E+09),
                        (8.592895630749110E+00, 8.347476619849094E+09),
                        (8.748744164861082E+00, 9.449408225536400E+09),
                        (8.907419309073834E+00, 1.063968931129575E+10),
                        (9.068972329347023E+00, 1.192223012961042E+10),
                        (9.233455421446157E+00, 1.330103063405093E+10),
                        (9.400921727806400E+00, 1.478017899089362E+10),
                        (9.571425354702223E+00, 1.636386713418054E+10),
                        (9.745021389728478E+00, 1.805632069655811E+10),
                        (9.921765919598508E+00, 1.986185162352030E+10),
                        (1.010171604826506E+01, 2.178498739329724E+10),
                        (1.028492991536986E+01, 2.383059077117696E+10),
                        (1.047146671502781E+01, 2.600394557413166E+10),
                        (1.066138671495187E+01, 2.831052231660086E+10),
                        (1.085475127592482E+01, 3.075591445374188E+10),
                        (1.105162287162409E+01, 3.334586323865519E+10),
                        (1.125206510880634E+01, 3.608625719039954E+10),
                        (1.145614274785792E+01, 3.898313232888238E+10),
                        (1.166392172371818E+01, 4.204267221721759E+10),
                        (1.187546916718221E+01, 4.527120748121231E+10),
                        (1.209085342658999E+01, 4.867521509408389E+10),
                        (1.231014408990886E+01, 5.226131731180368E+10),
                        (1.253341200721652E+01, 5.603628021734685E+10),
                        (1.276072931359183E+01, 6.000701184992552E+10),
                        (1.299216945242074E+01, 6.418055988630270E+10),
                        (1.322780719912494E+01, 6.856410883464980E+10),
                        (1.346771868532085E+01, 7.316497671211293E+10),
                        (1.371198142341682E+01, 7.799061123298459E+10),
                        (1.396067433165639E+01, 8.304858457910767E+10),
                        (1.421387775961582E+01, 8.834660675063837E+10),
                        (1.447167351416401E+01, 9.389244530988318E+10),
                        (1.473414488589332E+01, 9.969400090033122E+10),
                        (1.500137667602966E+01, 1.057596655485441E+11),
                        (1.527345522383077E+01, 1.120985019030221E+11),
                        (1.555046843448132E+01, 1.187201693037786E+11),
                        (1.583250580749397E+01, 1.256345411586507E+11),
                        (1.611965846562559E+01, 1.328516461551514E+11),
                        (1.641201918431785E+01, 1.403817643027305E+11),
                        (1.670968242167181E+01, 1.482353974265845E+11),
                        (1.701274434896617E+01, 1.564232688375824E+11),
                        (1.732130288172900E+01, 1.649563215766172E+11),
                        (1.763545771137303E+01, 1.738457161123761E+11),
                        (1.795531033740469E+01, 1.831028274354447E+11),
                        (1.828096410021736E+01, 1.927392414922294E+11),
                        (1.861252421447935E+01, 2.027667509085739E+11),
                        (1.895009780312740E+01, 2.131973499569905E+11),
                        (1.929379393197685E+01, 2.240432283487879E+11),
                        (1.964372364495940E+01, 2.353167666269465E+11),
                        (2.000000000000000E+01, 2.470305239319453E+11),
                        (2.032449182534652E+01, 2.579030738588072E+11),
                        (2.065424839792886E+01, 2.691480271670221E+11),
                        (2.098935513611907E+01, 2.807747384801088E+11),
                        (3.039220407997257E+01, 6.796211308075411E+11),
                        (3.088530516888327E+01, 7.039792802613912E+11),
                        (3.138640662141502E+01, 7.290406086763062E+11),
                        (3.189563824019757E+01, 7.548200868888253E+11),
                        (3.241313193385525E+01, 7.813332934130638E+11),
                        (3.293902175117595E+01, 8.085974863082582E+11),
                        (3.347344391583432E+01, 8.366309053351727E+11),
                        (3.401653686167849E+01, 8.654521075513728E+11),
                        (3.456844126858913E+01, 8.950799652840549E+11),
                        (3.512930009892055E+01, 9.255336170381827E+11),
                        (3.569925863453275E+01, 9.568324771641136E+11),
                        (3.627846451442459E+01, 9.889962854008933E+11),
                        (3.686706777297731E+01, 1.022045067254196E+12),
                        (3.746522087881866E+01, 1.055999127391694E+12),
                        (3.807307877431757E+01, 1.090879048249692E+12),
                        (3.869079891571956E+01, 1.126705682576751E+12),
                        (3.931854131393340E+01, 1.163500145429861E+12),
                        (3.995646857597943E+01, 1.201283805181227E+12),
                        (4.060474594711043E+01, 1.240078273439231E+12),
                        (4.126354135361589E+01, 1.279905393788630E+12),
                        (4.193302544632070E+01, 1.320787229165602E+12),
                        (4.261337164478962E+01, 1.362746048683358E+12),
                        (4.330475618224898E+01, 1.405804317787666E+12),
                        (4.400735815123716E+01, 1.449984610763401E+12),
                        (4.472135954999580E+01, 1.495309852275127E+12),
                        (4.544694532961358E+01, 1.541803183238278E+12),
                        (4.618430344193506E+01, 1.589486930525682E+12),
                        (4.693362488824660E+01, 1.638383854887015E+12),
                        (4.769510376875238E+01, 1.688518000476384E+12),
                        (4.846893733285307E+01, 1.739915210259069E+12),
                        (4.925532603024023E+01, 1.792603490304998E+12),
                        (5.005447356281974E+01, 1.846611522090996E+12),
                        (5.086658693747765E+01, 1.901968035708821E+12),
                        (5.169187651970211E+01, 1.958702283874122E+12),
                        (5.253055608807534E+01, 2.016843923516636E+12),
                        (5.338284288964969E+01, 2.076423020407534E+12),
                        (5.424895769622212E+01, 2.137470040757271E+12),
                        (5.512912486152176E+01, 2.200015841483303E+12),
                        (5.602357237932531E+01, 2.264091659470590E+12),
                        (5.693253194251530E+01, 2.329729099902661E+12),
                        (5.785623900309656E+01, 2.396960122383562E+12),
                        (5.879493283318652E+01, 2.465817025549761E+12),
                        (5.974885658699483E+01, 2.536332434001576E+12),
                        (6.071825736380887E+01, 2.608539177611441E+12),
                        (6.170338627200096E+01, 2.682470532751503E+12),
                        (6.270449849407409E+01, 2.758160005593318E+12),
                        (6.372185335276308E+01, 2.835641293988964E+12),
                        (6.475571437820813E+01, 2.914948305288780E+12),
                        (6.580634937621824E+01, 2.996115148698237E+12),
                        (6.687403049764221E+01, 3.079176169547209E+12),
                        (6.795903430886511E+01, 3.164165498683554E+12),
                        (6.906164186344861E+01, 3.251115108493706E+12),
                        (7.018213877493349E+01, 3.340061211157197E+12),
                        (7.132081529082352E+01, 3.431042634558869E+12),
                        (7.247796636776954E+01, 3.524100385845359E+12),
                        (7.365389174797360E+01, 3.619277046572504E+12),
                        (7.484889603683230E+01, 3.716616019429077E+12),
                        (7.606328878184047E+01, 3.816160633613701E+12),
                        (7.729738455277437E+01, 3.917954852232568E+12),
                        (7.855150302317644E+01, 4.022043280981642E+12),
                        (7.982596905316156E+01, 4.128471073746446E+12),
                        (8.112111277356729E+01, 4.237283917095765E+12),
                        (8.243726967146905E+01, 4.348528014378473E+12),
                        (8.377478067708293E+01, 4.462250066058021E+12),
                        (8.513399225207846E+01, 4.578497249082051E+12),
                        (8.651525647932409E+01, 4.697317193327824E+12),
                        (8.791893115408898E+01, 4.818757954981675E+12),
                        (8.934537987672422E+01, 4.942867990365911E+12),
                        (9.079497214684801E+01, 5.069696176500677E+12),
                        (9.226808345905883E+01, 5.199291698274241E+12),
                        (9.376509540020155E+01, 5.331703607006252E+12),
                        (9.528639574821162E+01, 5.466981866091986E+12),
                        (9.683237857256299E+01, 5.605176754904426E+12),
                        (9.840344433634577E+01, 5.746338321457387E+12),
                        (1.000000000000000E+02, 5.890515831457386E+12),
                        (1.023292992280754E+02, 6.102123472027861E+12),
                        (1.047128548050900E+02, 6.320153260446497E+12),
                        (1.071519305237607E+02, 6.544777365504612E+12),
                        (1.096478196143185E+02, 6.776177529939733E+12),
                        (1.122018454301964E+02, 7.014539423307548E+12),
                        (1.148153621496883E+02, 7.260048528444222E+12),
                        (1.174897554939529E+02, 7.512895666119430E+12),
                        (1.202264434617413E+02, 7.773274264248634E+12),
                        (1.230268770812381E+02, 8.041380699162122E+12),
                        (1.258925411794167E+02, 8.317414108122758E+12),
                        (1.288249551693134E+02, 8.601576189894241E+12),
                        (1.318256738556407E+02, 8.894070977816998E+12),
                        (1.348962882591654E+02, 9.195104567760781E+12),
                        (1.380384264602885E+02, 9.504884378581660E+12),
                        (1.412537544622755E+02, 9.823620354824418E+12),
                        (1.445439770745928E+02, 1.015152647499033E+13),
                        (1.479108388168208E+02, 1.048880185411277E+13),
                        (1.513561248436208E+02, 1.083564751788387E+13),
                        (1.548816618912481E+02, 1.119228668300965E+13),
                        (1.584893192461113E+02, 1.155897355935264E+13),
                        (1.621810097358930E+02, 1.193596744721025E+13),
                        (1.659586907437561E+02, 1.232353158020929E+13),
                        (1.698243652461744E+02, 1.272193365462608E+13),
                        (1.737800828749376E+02, 1.313144570155400E+13),
                        (1.778279410038923E+02, 1.355234389299316E+13),
                        (1.819700858609984E+02, 1.398490834709871E+13),
                        (1.862087136662868E+02, 1.442942288660350E+13),
                        (1.905460717963247E+02, 1.488617476491116E+13),
                        (1.949844599758045E+02, 1.535545435132147E+13),
                        (1.995262314968880E+02, 1.583755477245977E+13),
                        (2.041737944669529E+02, 1.633277150442027E+13),
                        (2.089296130854039E+02, 1.684140210369643E+13),
                        (2.137962089502232E+02, 1.736374790466818E+13),
                        (2.187761623949552E+02, 1.790008230229459E+13),
                        (2.238721138568339E+02, 1.845069627100483E+13),
                        (2.290867652767773E+02, 1.901590836015767E+13),
                        (2.344228815319922E+02, 1.959608075528396E+13),
                        (2.398832919019491E+02, 2.019157948213088E+13),
                        (2.454708915685030E+02, 2.080277449106095E+13),
                        (2.511886431509580E+02, 2.143004216963518E+13),
                        (2.570395782768864E+02, 2.207376477179097E+13),
                        (2.630267991895382E+02, 2.273433018560622E+13),
                        (2.691534803926916E+02, 2.341213167134843E+13),
                        (2.754228703338166E+02, 2.410756755601995E+13),
                        (2.818382931264454E+02, 2.482104083778808E+13),
                        (2.884031503126606E+02, 2.555296077489505E+13),
                        (2.951209226666385E+02, 2.630373383075783E+13),
                        (3.019951720402016E+02, 2.707377690332135E+13),
                        (3.090295432513590E+02, 2.786351236732911E+13),
                        (3.162277660168379E+02, 2.867335264633400E+13),
                        (3.235936569296283E+02, 2.950367516953141E+13),
                        (3.311311214825911E+02, 3.035487691947260E+13),
                        (3.388441561392026E+02, 3.122742573687622E+13),
                        (3.467368504525317E+02, 3.212182862969092E+13),
                        (3.548133892335754E+02, 3.303859120992604E+13),
                        (3.630780547701013E+02, 3.397822308225460E+13),
                        (3.715352290971725E+02, 3.494124660711691E+13),
                        (3.801893963205612E+02, 3.592819107253223E+13),
                        (3.890451449942806E+02, 3.693959355457624E+13),
                        (3.981071705534972E+02, 3.797599872442685E+13),
                        (4.073802778041127E+02, 3.903795838163673E+13),
                        (4.168693834703354E+02, 4.012603092525770E+13),
                        (4.265795188015927E+02, 4.124078103287737E+13),
                        (4.365158322401659E+02, 4.238277880323568E+13),
                        (4.466835921509631E+02, 4.355260589971910E+13),
                        (4.570881896148750E+02, 4.475083690894942E+13),
                        (4.677351412871982E+02, 4.597802895552939E+13),
                        (4.786300923226382E+02, 4.723471631834933E+13),
                        (4.897788193684462E+02, 4.852143756645389E+13),
                        (5.011872336272723E+02, 4.983887360730609E+13),
                        (5.128613839913648E+02, 5.118770381976145E+13),
                        (5.248074602497726E+02, 5.256861832530013E+13),
                        (5.370317963702527E+02, 5.398231816824654E+13),
                        (5.495408738576245E+02, 5.542951616202695E+13),
                        (5.623413251903492E+02, 5.691093628217921E+13),
                        (5.754399373371569E+02, 5.842731312825460E+13),
                        (5.888436553555889E+02, 5.997939199797330E+13),
                        (6.025595860743578E+02, 6.156792867003428E+13),
                        (6.165950018614822E+02, 6.319368891907341E+13),
                        (6.309573444801932E+02, 6.485744791903640E+13),
                        (6.456542290346555E+02, 6.655998972445104E+13),
                        (6.606934480075960E+02, 6.830213669979401E+13),
                        (6.760829753919818E+02, 7.008464740495280E+13),
                        (6.918309709189365E+02, 7.190824593524348E+13),
                        (7.079457843841378E+02, 7.377369231559439E+13),
                        (7.244359600749900E+02, 7.568178865360078E+13),
                        (7.413102413009175E+02, 7.763341520977069E+13),
                        (7.585775750291838E+02, 7.962950489514512E+13),
                        (7.762471166286916E+02, 8.167098951215694E+13),
                        (7.943282347242815E+02, 8.375881676465286E+13),
                        (8.128305161640992E+02, 8.589395016176288E+13),
                        (8.317637711026709E+02, 8.807736891993484E+13),
                        (8.511380382023765E+02, 9.031006771942873E+13),
                        (8.709635899560806E+02, 9.259305658994909E+13),
                        (8.912509381337455E+02, 9.492736093989847E+13),
                        (9.120108393559096E+02, 9.731402126633895E+13),
                        (9.332543007969911E+02, 9.975409462459869E+13),
                        (9.549925860214358E+02, 1.022486521448173E+14),
                        (9.772372209558107E+02, 1.047987783008963E+14),
                        (1.000000000000000E+03, 1.074055728110336E+14)], dtype=np.float)

        x = dat[:, 0]
        y = dat[:, 1]
        y_adj = min(y) * -1 + 1e6
        p = 3
        knots = fit.get_default_knots(p, x)
        bsp = fit.get_spline(p, knots, x, y)
        from matplotlib import pyplot as plt
        plt.loglog(x, y+y_adj, '*', label='data')
        pltx = np.linspace(x[0], x[-1], 2000)
        plt.loglog(pltx, bsp(pltx)+y_adj, label='cubic B-spline fit')
        plt.xlabel('density')
        plt.ylabel('$E_c$')
        plt.title('Adjusted energy cold curve log-log plot')
        plt.legend()
        plt.show(block=True)
















