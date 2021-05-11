# -*- coding: utf-8 -*-
'''
Гауссов импульс распространяется в одну сторону (TFSF boundary).
Источник находится в диэлектрике.
'''

import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

class GaussianDiff:
    '''
    Источник, создающий дифференцированный гауссов импульс
    '''

    def __init__(self, dg, wg, eps=1.0, mu=1.0, Sc=1.0, magnitude=1.0):
        '''
        magnitude - максимальное значение в источнике;
        dg - коэффициент, задающий начальную задержку гауссова импульса;
        wg - коэффициент, задающий ширину гауссова импульса.
        '''
        self.dg = dg
        self.wg = wg
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

    def getField(self, m, q):
        e = (q - m * numpy.sqrt(self.eps * self.mu) / self.Sc - self.dg) / self.wg
        return -2 * self.magnitude * e * numpy.exp(-(e ** 2))


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 400

    # Размер области моделирования вдоль оси X в метрах
    X = 1

    #Размер ячейки разбиения
    dx = 5e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)

    # Датчики для регистрации поля
    probesPos = [sourcePos + 50]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[:] = 3.0

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    source = GaussianDiff(30.0, 10.0, eps[sourcePos], mu[sourcePos])

    # Коэффициенты для расчета АВС второй степени
    # Sc' для правой границы согласно условию для правой границы
    
    Sc1Right = Sc / numpy.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = numpy.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = numpy.zeros(3)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия слева (PEC)
        Ez[0] = 0.0

        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 2 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    '''
    Спектр сигнала в датчике
    '''
    #Размер массива
    size = 4096

    #Шаг по частоте
    df = 1.0 / (size * dt)
    
    #Расчет спектра
    spectrum = numpy.abs(fft(probe.E, size))
    spectrum = fftshift(spectrum)
    f = numpy.arange(-size / 2 * df, size / 2 * df, df)

    #Построение графика
    plt.plot(f, spectrum /numpy.max(spectrum))
    plt.xlim(0, 7e9)
    plt.xlabel('Частота, Гц')
    plt.ylabel(r'$\frac{|P|}{P_{max}}$')
    plt.grid()
    plt.show()
