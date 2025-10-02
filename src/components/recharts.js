import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, Legend } from 'recharts';

function PitstopComparisonChart({ pitstopData }) {
    const scatterData = pitstopData.predicted.map((predicted, index) => ({
        lap: index + 1,
        predicted,
        actual: pitstopData.actual[index],
    }));

    return (
        <ScatterChart width={600} height={400}>
            <XAxis type="number" dataKey="lap" name="Lap" />
            <YAxis type="number" dataKey="predicted" name="Predicted Pitstop" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter name="Predicted" data={scatterData} fill="#8884d8" />
            <Scatter name="Actual" data={scatterData} dataKey="actual" fill="#82ca9d" />
        </ScatterChart>
    );
}
