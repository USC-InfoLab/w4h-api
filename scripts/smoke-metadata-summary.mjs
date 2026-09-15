/**
 * Smoke / unit checks for dataset-metadata-summary (no DB required for wear + shape).
 * Run: node scripts/smoke-metadata-summary.mjs
 */

import assert from 'assert';
import {
  aggregateWearCohort,
  buildMetadataSummary,
  mergeCollectionDateRanges,
  resolveScopeDates,
} from '../lib/dataset-metadata-summary.js';

function testAggregateWearCohort() {
  const nullResult = aggregateWearCohort(null);
  assert.strictEqual(nullResult, null);

  const empty = aggregateWearCohort({ subjects: [] });
  assert.strictEqual(empty, null);

  const source = {
    protocol: {
      daily_band_wear_percentage: 60,
      overall_band_wear_percentage: 60,
      daily_band_wear_hours: [['08:00', '20:00']],
    },
    subjects: [
      { subject_id: '1', is_overall_compliant: true, days_with_data: 7 },
      { subject_id: '2', is_overall_compliant: true, days_with_data: 5 },
      { subject_id: '3', is_overall_compliant: true, days_with_data: 9 },
      { subject_id: '4', is_overall_compliant: false, days_with_data: 3 },
      { subject_id: '5', is_overall_compliant: false, days_with_data: 6 },
    ],
    daily: [
      { subject_id: '1', compliance_percentage: 80 },
      { subject_id: '1', compliance_percentage: 70 },
      { subject_id: '2', compliance_percentage: 60 },
      { subject_id: '3', compliance_percentage: 90 },
      { subject_id: '4', compliance_percentage: 40 },
    ],
  };

  const cohort = aggregateWearCohort(source);
  assert.strictEqual(cohort.subjects_compliant, 3);
  assert.strictEqual(cohort.subjects_total, 5);
  assert.strictEqual(cohort.compliance_pct, 60);
  // median of [40, 60, 70, 80, 90] = 70
  assert.strictEqual(cohort.median_daily_wear_pct, 70);
  // median of days_with_data [3, 5, 6, 7, 9] = 6
  assert.strictEqual(cohort.median_days_worn, 6);

  const noDaily = aggregateWearCohort({
    subjects: [{ subject_id: '1', is_overall_compliant: true }],
  });
  assert.strictEqual(noDaily.median_daily_wear_pct, null);
  assert.strictEqual(noDaily.median_days_worn, null);
  assert.strictEqual(noDaily.subjects_compliant, 1);

  // days worn falls back to counting cached daily rows when days_with_data is absent
  const countedDays = aggregateWearCohort({
    subjects: [{ subject_id: '1' }, { subject_id: '2' }],
    daily: [
      { subject_id: '1', compliance_percentage: 80 },
      { subject_id: '1', compliance_percentage: 70 },
      { subject_id: '1', compliance_percentage: 65 },
      { subject_id: '2', compliance_percentage: 90 },
    ],
  });
  // median of [1, 3] = 2
  assert.strictEqual(countedDays.median_days_worn, 2);

  console.log('ok aggregateWearCohort');
}

function fitbitTables() {
  return [
    {
      name: 'heart_rate',
      description: 'Heart rate; non-zero values used as wear proxy',
      source_name: 'Fitbit',
      source_url: 'https://dev.fitbit.com/build/reference/web-api/',
      mapping_json: {
        signal_type: 'heart_rate',
        subject: 'id',
        time: { combine: ['date', 'timestamp'] },
        values: [
          {
            source: 'value',
            name: 'bpm',
            type: 'integer',
            description: 'Heart rate in beats per minute',
          },
        ],
      },
      physical_schema: 'ds_sample',
      physical_table: 'fitbit_heart_rate',
    },
    {
      name: 'calories',
      description: 'Calories',
      source_name: 'Fitbit',
      source_url: null,
      mapping_json: {
        signal_type: 'calories',
        subject: 'id',
        time: { combine: ['date', 'time'] },
        values: [
          { source: 'value', name: 'calories', type: 'double precision', description: 'kcal' },
        ],
      },
      physical_schema: 'ds_sample',
      physical_table: 'fitbit_calories',
    },
    {
      name: 'subjects',
      description: 'Enrollment',
      source_name: 'Fitbit',
      mapping_json: {
        signal_type: 'subjects',
        subject: 'id',
        values: [],
      },
      physical_schema: 'ds_sample',
      physical_table: 'fitbit_subjects',
    },
  ];
}

function icdeTables() {
  return [
    {
      name: 'heart_rate',
      description: 'HR',
      source_name: null,
      mapping_json: {
        signal_type: 'heart_rate',
        sampling_interval: '5min',
        subject: 'id',
        time: { combine: ['date', 'timestamp'] },
        values: [
          {
            source: 'value',
            name: 'heart_rate_bpm',
            type: 'double precision',
            description: 'Heart rate',
            unit: 'bpm',
          },
        ],
      },
      physical_schema: 'ds_icde',
      physical_table: 'icde_heart_rate',
    },
    {
      name: 'calories',
      mapping_json: {
        signal_type: 'calories',
        sampling_interval: '5min',
        subject: 'id',
        time: { combine: ['date', 'timestamp'] },
        values: [
          {
            source: 'value',
            name: 'calories',
            type: 'double precision',
            unit: 'kcal',
          },
        ],
      },
      physical_schema: 'ds_icde',
      physical_table: 'icde_calories',
    },
  ];
}

function testBuildFitbitShape() {
  const summary = buildMetadataSummary(
    {
      origin: 'W4H Samples',
      description: 'Fitbit sample',
      device_type: 'wearable',
      date_start: '2018-06-01',
      date_end: '2018-11-30',
      subject_count: 5,
      irb_reference: null,
      read_only: true,
      table_count: 4,
      summary: {
        observations: 1400000,
        observations_approx: true,
        wear_compliance: {
          protocol: {
            daily_band_wear_percentage: 60,
            overall_band_wear_percentage: 60,
            daily_band_wear_hours: [['08:00', '20:00']],
          },
          subjects: [
            { subject_id: '1', is_overall_compliant: true },
            { subject_id: '2', is_overall_compliant: false },
          ],
          daily: [
            { compliance_percentage: 70 },
            { compliance_percentage: 50 },
          ],
        },
      },
    },
    fitbitTables(),
    new Map([
      [
        'heart_rate',
        {
          mode_minutes: 1,
          mode_count: 100,
          mode_pct: 95,
          max_gap_minutes: 11,
          gaps_sampled: 105,
          approx: false,
        },
      ],
    ])
  );

  assert.ok(summary.scope);
  assert.ok(summary.subjects);
  assert.strictEqual(summary.subjects.enrolled_participants, 5);
  assert.strictEqual(summary.subjects.participant_age, null);

  const fromSubjectsTable = buildMetadataSummary(
    { subject_count: 99, table_count: 4 },
    fitbitTables(),
    null,
    null,
    null,
    5
  );
  assert.strictEqual(fromSubjectsTable.subjects.enrolled_participants, 5);
  assert.ok(summary.device);
  assert.ok(summary.cadence);
  assert.strictEqual(summary.temporal, undefined);
  assert.ok(summary.provenance);
  assert.ok(summary.compliance);
  assert.ok(summary.semantic);
  assert.strictEqual(summary.structural, undefined);
  assert.strictEqual(summary.contextual, undefined);

  assert.strictEqual(summary.device.vendor, 'Fitbit');
  assert.ok(summary.device.signal_types.includes('heart_rate'));
  assert.ok(!summary.device.signal_types.includes('subjects'));
  assert.ok(!summary.cadence.signals.some((s) => s.signal === 'subjects'));
  assert.ok(!summary.semantic.signals.some((s) => s.signal === 'subjects'));
  assert.strictEqual(summary.scope.table_count, 4);
  assert.strictEqual(summary.scope.observations, 1400000);
  assert.strictEqual(summary.scope.study_length_days, 183);
  assert.strictEqual(summary.cadence.dataset, undefined);
  const hrCadence = summary.cadence.signals.find((s) => s.signal === 'heart_rate');
  assert.ok(hrCadence?.description);
  assert.strictEqual(hrCadence?.declared_interval, null);
  assert.strictEqual(hrCadence?.observed_cadence?.mode_minutes, 1);
  assert.strictEqual(summary.compliance.wear_cohort.subjects_compliant, 1);
  assert.strictEqual(summary.compliance.wear_cohort.subjects_total, 2);
  assert.strictEqual(summary.compliance.wear_cohort.median_daily_wear_pct, 60);
  assert.strictEqual(summary.provenance.ingestion_source, null);
  assert.strictEqual(summary.provenance.package_vendor, null);
  assert.strictEqual(summary.device.model, null);

  console.log('ok buildMetadataSummary Fitbit shape');
}

function testIngestionSourceFromRun() {
  const withRun = buildMetadataSummary(
    { table_count: 4, summary: {} },
    fitbitTables(),
    null,
    { slug: 'w4h-sample-fitbit-csv-importer', label: 'Fitbit CSV importer' }
  );
  assert.strictEqual(withRun.provenance.ingestion_source, 'Fitbit CSV importer');
  assert.strictEqual(withRun.provenance.package_vendor, null);

  console.log('ok ingestion source from contribution run');
}

function testCollectionDateRange() {
  const merged = mergeCollectionDateRanges([
    { date_start: '2018-06-09', date_end: '2018-06-09' },
    { date_start: '2012-06-23', date_end: '2016-08-20' },
    { date_start: '2021-03-07', date_end: '2021-06-05' },
  ]);
  assert.strictEqual(merged.date_start, '2012-06-23');
  assert.strictEqual(merged.date_end, '2021-06-05');

  const catalogWins = resolveScopeDates(
    { date_start: '2019-01-01', date_end: null, summary: {} },
    { date_start: '2012-06-23', date_end: '2021-06-05' }
  );
  assert.strictEqual(catalogWins.date_start, '2019-01-01');
  assert.strictEqual(catalogWins.date_end, '2021-06-05');

  const fromData = buildMetadataSummary(
    { table_count: 4, summary: {} },
    fitbitTables(),
    null,
    null,
    { date_start: '2012-06-23', date_end: '2021-06-05' }
  );
  assert.strictEqual(fromData.scope.date_start, '2012-06-23');
  assert.strictEqual(fromData.scope.date_end, '2021-06-05');
  assert.ok(fromData.scope.study_length_days > 3000);

  console.log('ok collection date range');
}

function testBuildIcdeShape() {
  const summary = buildMetadataSummary(
    {
      origin: 'W4H Samples',
      description: 'ICDE demo',
      date_start: '2020-01-01',
      date_end: '2020-01-10',
      subject_count: 3,
      read_only: true,
    },
    icdeTables()
  );

  const hr = summary.cadence.signals.find((s) => s.signal === 'heart_rate');
  assert.strictEqual(hr.declared_interval, '5min');
  assert.strictEqual(hr.native_sensor_rate, null);
  assert.strictEqual(hr.observed_cadence, null);

  const calVars = summary.semantic.signals.find((s) => s.signal === 'calories')?.variables;
  assert.strictEqual(calVars?.[0]?.unit, 'kcal');
  assert.strictEqual(calVars?.[0]?.ontology_code, null);
  assert.strictEqual(summary.compliance.wear_cohort, null);

  console.log('ok buildMetadataSummary ICDE shape');
}

testAggregateWearCohort();
testBuildFitbitShape();
testIngestionSourceFromRun();
testCollectionDateRange();
testBuildIcdeShape();
console.log('All smoke checks passed.');
